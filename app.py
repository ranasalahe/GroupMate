"""GroupMate — AI-powered task distribution for student group projects.

Core loop: create a group -> AI generates skill tags from the project
description -> members pick their strong suits -> once everyone has
submitted, AI distributes tasks fairly -> members track progress on a
shared dashboard.
"""

import json
import os
from datetime import datetime, timezone

import gradio as gr
from dateutil import parser as date_parser
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

OPENAI_MODEL = "gpt-4o"
FALLING_BEHIND_THRESHOLD = 0.2  # a member is "behind" if their completion
# fraction trails the group's time-elapsed fraction by more than this.
MAX_DESCRIPTION_CHARS = 4000
OPENAI_JSON_RETRIES = 2  # extra attempts if the model returns malformed JSON

_supabase_client = None
_openai_client = None


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


# ---------------------------------------------------------------------------
# AI calls
# ---------------------------------------------------------------------------

def _call_openai_json(system_prompt: str, user_prompt: str) -> dict:
    """Call OpenAI in JSON mode, retrying if the model returns malformed JSON.

    The model is asked to emit JSON and (with response_format=json_object)
    is constrained to produce syntactically valid JSON, but can still omit
    expected keys or wrap values unexpectedly — retrying a couple of times
    is cheap insurance against an occasional bad sample.
    """
    last_error = None
    for _ in range(OPENAI_JSON_RETRIES + 1):
        response = get_openai().chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise RuntimeError("OpenAI did not return valid JSON after retrying") from last_error


def generate_tags(description: str) -> list[str]:
    """Ask the model for 6-10 project-specific skill tags."""
    data = _call_openai_json(
        system_prompt=(
            "You generate skill tags for student group projects. "
            "Given a project description, respond ONLY with a JSON "
            'object of the form {"tags": ["tag1", "tag2", ...]} '
            "containing 6 to 10 tags. Tags must be specific to what "
            "this project actually needs (e.g. 'React frontend', "
            "'survey design', 'financial modeling'), not generic "
            "soft skills like 'teamwork' or 'communication'."
        ),
        user_prompt=description,
    )
    return list(data["tags"])


def ai_distribute_tasks(description: str, members: list[dict]) -> list[dict]:
    """Ask the model to break the project into tasks and assign them fairly.

    `members` is a list of {"name": str, "strong_suits": [str, ...]}.
    Returns a list of {"title", "estimated_hours", "assigned_to",
    "shared", "shared_with"}.
    """
    member_summary = "\n".join(
        f"- {m['name']}: {', '.join(m['strong_suits']) or 'no tags selected'}"
        for m in members
    )
    data = _call_openai_json(
        system_prompt=(
            "You split a student group project into concrete tasks "
            "and assign them fairly across members based on skill-tag "
            "match and balanced total workload (estimated_hours per "
            "member should be roughly even). If a task is large "
            "enough that splitting it across multiple members is "
            "more efficient, mark it shared and list every member "
            "working on it in shared_with; otherwise shared_with is "
            "empty and assigned_to names the single owner. Respond "
            "ONLY with a JSON object of the form "
            '{"tasks": [{"title": str, "estimated_hours": number, '
            '"assigned_to": str, "shared": bool, '
            '"shared_with": [str, ...]}, ...]}. '
            "Every member name used must exactly match one of the "
            "names given."
        ),
        user_prompt=(
            f"Project description:\n{description}\n\n"
            f"Members and their strong suits:\n{member_summary}"
        ),
    )
    return list(data["tasks"])


# ---------------------------------------------------------------------------
# Group / member operations
# ---------------------------------------------------------------------------

def create_group(name: str, description: str, deadline_str: str):
    if not name.strip() or not description.strip() or not deadline_str.strip():
        return "", "Please fill in name, description, and deadline.", []

    if len(description) > MAX_DESCRIPTION_CHARS:
        return "", f"Description is too long (max {MAX_DESCRIPTION_CHARS} characters).", []

    try:
        deadline = date_parser.parse(deadline_str)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return "", "Could not parse that deadline. Try e.g. 2026-09-20 18:00.", []

    if deadline <= datetime.now(timezone.utc):
        return "", "Deadline must be in the future.", []

    try:
        tags = generate_tags(description)

        db = get_supabase()
        result = (
            db.table("groups")
            .insert(
                {
                    "name": name.strip(),
                    "description": description.strip(),
                    "deadline": deadline.isoformat(),
                    "tags": tags,
                }
            )
            .execute()
        )
    except Exception as exc:  # AI/network failures shouldn't crash the app
        return "", f"Could not create the group: {exc}", []

    group_id = result.data[0]["id"]
    status = f"Group '{name}' created. Share this Group ID with your teammates: {group_id}"
    return group_id, status, tags


def add_member(group_id: str, member_name: str):
    if not group_id.strip() or not member_name.strip():
        return "Please provide a Group ID and a member name."

    try:
        db = get_supabase()
        group = db.table("groups").select("id").eq("id", group_id.strip()).execute()
        if not group.data:
            return f"No group found with ID {group_id}."

        db.table("members").insert(
            {"group_id": group_id.strip(), "name": member_name.strip()}
        ).execute()
    except Exception as exc:
        return f"Could not add member: {exc}"
    return f"Added '{member_name}' to the group."


def load_group_tags(group_id: str):
    if not group_id.strip():
        return gr.CheckboxGroup(choices=[]), "Please provide a Group ID."

    try:
        db = get_supabase()
        group = db.table("groups").select("tags").eq("id", group_id.strip()).execute()
    except Exception as exc:
        return gr.CheckboxGroup(choices=[]), f"Could not load tags: {exc}"

    if not group.data:
        return gr.CheckboxGroup(choices=[]), "No group found with that ID."
    tags = group.data[0]["tags"]
    return gr.CheckboxGroup(choices=tags), f"Loaded {len(tags)} tags."


def submit_strong_suits(group_id: str, member_name: str, selected_tags: list[str]):
    if not group_id.strip() or not member_name.strip():
        return "Please provide a Group ID and your member name."

    try:
        db = get_supabase()
        member = (
            db.table("members")
            .select("id")
            .eq("group_id", group_id.strip())
            .ilike("name", member_name.strip())
            .execute()
        )
        if not member.data:
            return f"No member named '{member_name}' found in that group."

        db.table("members").update(
            {"strong_suits": selected_tags, "submitted": True}
        ).eq("id", member.data[0]["id"]).execute()

        status = f"Saved strong suits for {member_name}."

        all_members = (
            db.table("members").select("submitted").eq("group_id", group_id.strip()).execute()
        )
        if all_members.data and all(m["submitted"] for m in all_members.data):
            distribution_status = distribute_tasks(group_id.strip())
            status += f" Everyone has submitted — {distribution_status}"
    except Exception as exc:
        return f"Could not save strong suits: {exc}"

    return status


# ---------------------------------------------------------------------------
# Task distribution
# ---------------------------------------------------------------------------

def distribute_tasks(group_id: str) -> str:
    try:
        return _distribute_tasks(group_id)
    except Exception as exc:
        return f"Could not distribute tasks: {exc}"


def _distribute_tasks(group_id: str) -> str:
    db = get_supabase()

    existing = db.table("tasks").select("id").eq("group_id", group_id).execute()
    if existing.data:
        return "Tasks were already distributed for this group."

    group = db.table("groups").select("description").eq("id", group_id).execute()
    if not group.data:
        return "No group found with that ID."
    description = group.data[0]["description"]

    members_result = (
        db.table("members").select("id, name, strong_suits").eq("group_id", group_id).execute()
    )
    members = members_result.data
    if not members:
        return "No members found for this group."

    ai_tasks = ai_distribute_tasks(
        description,
        [{"name": m["name"], "strong_suits": m["strong_suits"]} for m in members],
    )

    name_to_id = {m["name"].strip().lower(): m["id"] for m in members}

    rows = []
    for task in ai_tasks:
        owner_id = name_to_id.get(str(task.get("assigned_to", "")).strip().lower())
        shared_with_ids = [
            name_to_id[n.strip().lower()]
            for n in task.get("shared_with", [])
            if n.strip().lower() in name_to_id
        ]
        rows.append(
            {
                "group_id": group_id,
                "title": task["title"],
                "estimated_hours": task.get("estimated_hours", 1),
                "assigned_to": owner_id,
                "shared": bool(task.get("shared", False)),
                "shared_with": shared_with_ids,
            }
        )

    db.table("tasks").insert(rows).execute()
    return f"Distributed {len(rows)} tasks across {len(members)} members."


# ---------------------------------------------------------------------------
# Dashboard / progress
# ---------------------------------------------------------------------------

def get_dashboard(group_id: str):
    if not group_id.strip():
        return "Please provide a Group ID.", None, "—"

    try:
        return _get_dashboard(group_id)
    except Exception as exc:
        return f"Could not load dashboard: {exc}", None, "—"


def _get_dashboard(group_id: str):
    db = get_supabase()
    group = db.table("groups").select("name, deadline").eq("id", group_id.strip()).execute()
    if not group.data:
        return "No group found with that ID.", None, "—"

    deadline = date_parser.parse(group.data[0]["deadline"])
    now = datetime.now(timezone.utc)
    remaining = deadline - now
    if remaining.total_seconds() > 0:
        countdown = f"{remaining.days}d {remaining.seconds // 3600}h remaining"
    else:
        countdown = "Deadline has passed"

    members = db.table("members").select("id, name").eq("group_id", group_id.strip()).execute().data
    tasks = db.table("tasks").select("*").eq("group_id", group_id.strip()).execute().data

    if not tasks:
        return "No tasks yet — waiting on task distribution.", None, countdown

    created_at = date_parser.parse(
        db.table("groups").select("created_at").eq("id", group_id.strip()).execute().data[0]["created_at"]
    )
    total_span = (deadline - created_at).total_seconds()
    elapsed_fraction = (
        min(max((now - created_at).total_seconds() / total_span, 0), 1) if total_span > 0 else 1
    )

    rows = []
    total_tasks = 0
    total_done = 0
    for member in members:
        member_tasks = [
            t
            for t in tasks
            if t["assigned_to"] == member["id"] or member["id"] in (t["shared_with"] or [])
        ]
        done = sum(1 for t in member_tasks if t["completed"])
        count = len(member_tasks)
        total_tasks += count
        total_done += done
        completion_fraction = (done / count) if count else 1.0
        behind = "Yes" if (elapsed_fraction - completion_fraction) > FALLING_BEHIND_THRESHOLD else "No"
        rows.append(
            [member["name"], f"{done}/{count}", f"{round(completion_fraction * 100)}%", behind]
        )

    group_progress = f"{round((total_done / total_tasks) * 100) if total_tasks else 0}% of tasks complete"
    return group_progress, rows, countdown


def get_my_tasks(group_id: str, member_name: str):
    if not group_id.strip() or not member_name.strip():
        return gr.CheckboxGroup(choices=[], value=[]), "Please provide a Group ID and your name."

    try:
        return _get_my_tasks(group_id, member_name)
    except Exception as exc:
        return gr.CheckboxGroup(choices=[], value=[]), f"Could not load tasks: {exc}"


def _get_my_tasks(group_id: str, member_name: str):
    db = get_supabase()
    member = (
        db.table("members")
        .select("id")
        .eq("group_id", group_id.strip())
        .ilike("name", member_name.strip())
        .execute()
    )
    if not member.data:
        return gr.CheckboxGroup(choices=[], value=[]), "No member found with that name in that group."

    member_id = member.data[0]["id"]
    tasks = db.table("tasks").select("*").eq("group_id", group_id.strip()).execute().data
    my_tasks = [
        t for t in tasks if t["assigned_to"] == member_id or member_id in (t["shared_with"] or [])
    ]

    if not my_tasks:
        return gr.CheckboxGroup(choices=[], value=[]), "No tasks assigned yet."

    choices = []
    completed = []
    for t in my_tasks:
        label = f"{t['title']} (~{t['estimated_hours']}h)" + (" [shared]" if t["shared"] else "")
        choices.append((label, t["id"]))
        if t["completed"]:
            completed.append(t["id"])

    return gr.CheckboxGroup(choices=choices, value=completed), f"Loaded {len(my_tasks)} tasks."


def save_task_updates(group_id: str, member_name: str, completed_task_ids: list[str]):
    if not group_id.strip():
        return "Please provide a Group ID."

    try:
        db = get_supabase()
        tasks = db.table("tasks").select("id").eq("group_id", group_id.strip()).execute().data
        all_ids = {t["id"] for t in tasks}
        completed_set = set(completed_task_ids)

        for task_id in all_ids:
            db.table("tasks").update({"completed": task_id in completed_set}).eq("id", task_id).execute()
    except Exception as exc:
        return f"Could not save task updates: {exc}"

    return "Saved. Refresh the Dashboard tab to see updated progress."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CREAM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.stone,
).set(
    body_background_fill="#FBF4E4",
    body_background_fill_dark="#FBF4E4",
    background_fill_primary="#FBF4E4",
    background_fill_primary_dark="#FBF4E4",
    background_fill_secondary="#F3E9D2",
    background_fill_secondary_dark="#F3E9D2",
    block_background_fill="#FFFDF7",
    block_background_fill_dark="#FFFDF7",
    block_border_color="#E1D3B0",
    block_border_color_dark="#E1D3B0",
    border_color_primary="#E1D3B0",
    border_color_primary_dark="#E1D3B0",
    body_text_color="#3B3327",
    body_text_color_dark="#3B3327",
    body_text_color_subdued="#7A6F5C",
    body_text_color_subdued_dark="#7A6F5C",
    button_primary_background_fill="#4B7A56",
    button_primary_background_fill_hover="#3D6446",
    button_primary_background_fill_dark="#4B7A56",
    button_primary_text_color="#FFFFFF",
    input_background_fill="#FFFDF7",
    input_background_fill_dark="#FFFDF7",
    input_border_color="#E1D3B0",
    input_border_color_dark="#E1D3B0",
)

with gr.Blocks(title="GroupMate", theme=CREAM_THEME) as demo:
    gr.Markdown("# GroupMate\nFair, AI-powered task distribution for group projects.")

    with gr.Tab("1. Create Group"):
        cg_name = gr.Textbox(label="Project / Group Name")
        cg_description = gr.Textbox(label="Project Description", lines=5)
        cg_deadline = gr.Textbox(label="Deadline", placeholder="2026-09-20 18:00")
        cg_button = gr.Button("Create Group & Generate Tags", variant="primary")
        cg_group_id = gr.Textbox(label="Group ID (share this with your team)", interactive=False)
        cg_status = gr.Markdown()
        cg_tags = gr.JSON(label="AI-generated skill tags")
        cg_button.click(
            create_group,
            inputs=[cg_name, cg_description, cg_deadline],
            outputs=[cg_group_id, cg_status, cg_tags],
        )

    with gr.Tab("2. Add Members"):
        am_group_id = gr.Textbox(label="Group ID")
        am_name = gr.Textbox(label="Member Name")
        am_button = gr.Button("Add Member", variant="primary")
        am_status = gr.Markdown()
        am_button.click(add_member, inputs=[am_group_id, am_name], outputs=am_status)

    with gr.Tab("3. Strong Suits"):
        ss_group_id = gr.Textbox(label="Group ID")
        ss_load_button = gr.Button("Load Available Tags")
        ss_load_status = gr.Markdown()
        ss_name = gr.Textbox(label="Your Name")
        ss_tags = gr.CheckboxGroup(choices=[], label="Select your strong suits")
        ss_submit = gr.Button("Submit Strong Suits", variant="primary")
        ss_status = gr.Markdown()
        ss_load_button.click(load_group_tags, inputs=ss_group_id, outputs=[ss_tags, ss_load_status])
        ss_submit.click(
            submit_strong_suits, inputs=[ss_group_id, ss_name, ss_tags], outputs=ss_status
        )

    with gr.Tab("4. Dashboard"):
        db_group_id = gr.Textbox(label="Group ID")
        db_refresh = gr.Button("Refresh Dashboard", variant="primary")
        db_countdown = gr.Markdown()
        db_progress = gr.Markdown()
        db_table = gr.Dataframe(
            headers=["Member", "Tasks Done", "Completion %", "Falling Behind?"],
            label="Per-member progress",
        )
        db_refresh.click(
            get_dashboard, inputs=db_group_id, outputs=[db_progress, db_table, db_countdown]
        )

    with gr.Tab("5. My Tasks"):
        mt_group_id = gr.Textbox(label="Group ID")
        mt_name = gr.Textbox(label="Your Name")
        mt_load = gr.Button("Load My Tasks")
        mt_status = gr.Markdown()
        mt_tasks = gr.CheckboxGroup(choices=[], label="Check off completed tasks")
        mt_save = gr.Button("Save Task Updates", variant="primary")
        mt_load.click(get_my_tasks, inputs=[mt_group_id, mt_name], outputs=[mt_tasks, mt_status])
        mt_save.click(
            save_task_updates, inputs=[mt_group_id, mt_name, mt_tasks], outputs=mt_status
        )


if __name__ == "__main__":
    demo.launch()
