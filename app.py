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

CREAM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
).set(
    body_background_fill="#F6EEDD",
    body_background_fill_dark="#F6EEDD",
    background_fill_primary="#F6EEDD",
    background_fill_primary_dark="#F6EEDD",
    background_fill_secondary="#EAE0C7",
    background_fill_secondary_dark="#EAE0C7",
    block_background_fill="#FCF8EF",
    block_background_fill_dark="#FCF8EF",
    block_border_color="#DCCEA8",
    block_border_color_dark="#DCCEA8",
    block_label_background_fill="#EAE0C7",
    block_label_background_fill_dark="#EAE0C7",
    block_label_text_color="#1A1A1A",
    block_label_text_color_dark="#1A1A1A",
    block_title_text_color="#1A1A1A",
    block_title_text_color_dark="#1A1A1A",
    border_color_primary="#DCCEA8",
    border_color_primary_dark="#DCCEA8",
    body_text_color="#1A1A1A",
    body_text_color_dark="#1A1A1A",
    body_text_color_subdued="#5C5342",
    body_text_color_subdued_dark="#5C5342",
    button_primary_background_fill="#DCCEA8",
    button_primary_background_fill_hover="#CBB98A",
    button_primary_background_fill_dark="#DCCEA8",
    button_primary_text_color="#1A1A1A",
    button_primary_text_color_dark="#1A1A1A",
    button_secondary_background_fill="#EAE0C7",
    button_secondary_background_fill_hover="#DCCEA8",
    button_secondary_background_fill_dark="#EAE0C7",
    button_secondary_text_color="#1A1A1A",
    button_secondary_text_color_dark="#1A1A1A",
    input_background_fill="#FCF8EF",
    input_background_fill_dark="#FCF8EF",
    input_border_color="#DCCEA8",
    input_border_color_dark="#DCCEA8",
    checkbox_background_color="#FCF8EF",
    checkbox_background_color_dark="#FCF8EF",
    checkbox_background_color_selected="#CBB98A",
    checkbox_background_color_selected_dark="#CBB98A",
    checkbox_border_color="#DCCEA8",
    checkbox_border_color_dark="#DCCEA8",
    checkbox_label_background_fill="#FCF8EF",
    checkbox_label_background_fill_dark="#FCF8EF",
    checkbox_label_background_fill_selected="#EAE0C7",
    checkbox_label_background_fill_selected_dark="#EAE0C7",
    checkbox_label_text_color="#1A1A1A",
    checkbox_label_text_color_dark="#1A1A1A",
    checkbox_label_text_color_selected="#1A1A1A",
    checkbox_label_text_color_selected_dark="#1A1A1A",
    slider_color="#CBB98A",
    slider_color_dark="#CBB98A",
    table_even_background_fill="#FCF8EF",
    table_even_background_fill_dark="#FCF8EF",
    table_odd_background_fill="#EAE0C7",
    table_odd_background_fill_dark="#EAE0C7",
    table_border_color="#DCCEA8",
    table_border_color_dark="#DCCEA8",
    table_row_focus="#DCCEA8",
    table_row_focus_dark="#DCCEA8",
    color_accent="#CBB98A",
    color_accent_soft="#EAE0C7",
    color_accent_soft_dark="#EAE0C7",
    border_color_accent="#DCCEA8",
    border_color_accent_dark="#DCCEA8",
    link_text_color="#1A1A1A",
    link_text_color_dark="#1A1A1A",
    link_text_color_hover="#5C5342",
    link_text_color_hover_dark="#5C5342",
)

RESPONSIVE_CSS = """
.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
}
@media (max-width: 640px) {
    .gradio-container { padding: 6px !important; }
    .gr-button { font-size: 1rem !important; padding: 10px !important; }
    h1 { font-size: 1.4rem !important; }
}
.step-nav-row {
    flex-wrap: wrap !important;
}
"""

def create_group_ui(name: str, description: str, deadline_str: str):
    group_id, status, tags = create_group(name, description, deadline_str)
    tags_md = "\n".join(f"- {t}" for t in tags) if tags else ""
    stay_here = gr.update(visible=True)
    move_on = gr.update(visible=bool(group_id))
    return group_id, status, tags_md, gr.update(visible=not group_id), move_on


def _advance():
    """Generic step transition: hide the current step, show the next one."""
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(title="GroupMate") as demo:
    gr.Markdown("# GroupMate\nFair, AI-powered task distribution for group projects.")

    with gr.Group():
        shared_group_id = gr.Textbox(
            label="Group ID",
            placeholder="Paste your team's Group ID here if you're joining an existing group",
            buttons=["copy"],
        )
        shared_name = gr.Textbox(
            label="Your Name",
            placeholder="Your name, exactly as added in Add Members",
            info="Used for Strong Suits and My Tasks.",
        )

    # --- Step 1: Create Group ---------------------------------------------
    with gr.Column(visible=True) as step_create:
        gr.Markdown("## Create Group")
        gr.Markdown("Start a new project. This generates skill tags and a Group ID to share.")
        cg_name = gr.Textbox(label="Project / Group Name")
        cg_description = gr.Textbox(label="Project Description", lines=5)
        cg_deadline = gr.Textbox(label="Deadline", placeholder="2026-09-20 18:00")
        cg_button = gr.Button("Create Group & Continue →", variant="primary")
        cg_status = gr.Markdown()
        cg_tags = gr.Markdown()
        cg_skip = gr.Button("I already have a Group ID →", variant="secondary")

    # --- Step 2: Add Members ------------------------------------------------
    with gr.Column(visible=False) as step_members:
        gr.Markdown("## Add Members")
        gr.Markdown("Add each teammate by name. Repeat for everyone in the group.")
        am_name = gr.Textbox(label="New Member's Name")
        am_button = gr.Button("Add Member", variant="primary")
        am_status = gr.Markdown()
        with gr.Row(elem_classes="step-nav-row"):
            am_back = gr.Button("← Back", variant="secondary")
            am_next = gr.Button("Continue to Strong Suits →", variant="primary")

    # --- Step 3: Strong Suits ------------------------------------------------
    with gr.Column(visible=False) as step_suits:
        gr.Markdown("## Strong Suits")
        gr.Markdown(
            "Load the group's skill tags, then pick the ones that match your "
            "strengths. Once everyone submits, tasks are distributed automatically."
        )
        ss_load_button = gr.Button("Load Available Tags")
        ss_load_status = gr.Markdown()
        ss_tags = gr.CheckboxGroup(choices=[], label="Select your strong suits")
        ss_submit = gr.Button("Submit Strong Suits", variant="primary")
        ss_status = gr.Markdown()
        with gr.Row(elem_classes="step-nav-row"):
            ss_back = gr.Button("← Back", variant="secondary")
            ss_next = gr.Button("Continue to Dashboard →", variant="primary")

    # --- Step 4: Dashboard ------------------------------------------------
    with gr.Column(visible=False) as step_dashboard:
        gr.Markdown("## Dashboard")
        gr.Markdown("See group and per-member progress, and who's falling behind.")
        db_refresh = gr.Button("Refresh Dashboard", variant="primary")
        db_countdown = gr.Markdown()
        db_progress = gr.Markdown()
        db_table = gr.Dataframe(
            headers=["Member", "Tasks Done", "Completion %", "Falling Behind?"],
            label="Per-member progress",
        )
        with gr.Row(elem_classes="step-nav-row"):
            db_back = gr.Button("← Back", variant="secondary")
            db_next = gr.Button("Continue to My Tasks →", variant="primary")

    # --- Step 5: My Tasks ------------------------------------------------
    with gr.Column(visible=False) as step_tasks:
        gr.Markdown("## My Tasks")
        gr.Markdown("Load your assigned tasks and check them off as you complete them.")
        mt_load = gr.Button("Load My Tasks")
        mt_status = gr.Markdown()
        mt_tasks = gr.CheckboxGroup(choices=[], label="Check off completed tasks")
        mt_save = gr.Button("Save Task Updates", variant="primary")
        mt_back = gr.Button("← Back to Dashboard", variant="secondary")

    # --- Wiring: step content ------------------------------------------------
    cg_button.click(
        create_group_ui,
        inputs=[cg_name, cg_description, cg_deadline],
        outputs=[shared_group_id, cg_status, cg_tags, step_create, step_members],
    )
    am_button.click(add_member, inputs=[shared_group_id, am_name], outputs=am_status)
    ss_load_button.click(
        load_group_tags, inputs=shared_group_id, outputs=[ss_tags, ss_load_status]
    )
    ss_submit.click(
        submit_strong_suits, inputs=[shared_group_id, shared_name, ss_tags], outputs=ss_status
    )
    db_refresh.click(
        get_dashboard, inputs=shared_group_id, outputs=[db_progress, db_table, db_countdown]
    )
    mt_load.click(
        get_my_tasks, inputs=[shared_group_id, shared_name], outputs=[mt_tasks, mt_status]
    )
    mt_save.click(
        save_task_updates, inputs=[shared_group_id, shared_name, mt_tasks], outputs=mt_status
    )

    # --- Wiring: step navigation ------------------------------------------------
    cg_skip.click(_advance, outputs=[step_create, step_members])
    am_next.click(_advance, outputs=[step_members, step_suits])
    ss_next.click(_advance, outputs=[step_suits, step_dashboard])
    db_next.click(_advance, outputs=[step_dashboard, step_tasks])
    am_back.click(_advance, outputs=[step_members, step_create])
    ss_back.click(_advance, outputs=[step_suits, step_members])
    db_back.click(_advance, outputs=[step_dashboard, step_suits])
    mt_back.click(_advance, outputs=[step_tasks, step_dashboard])


if __name__ == "__main__":
    demo.launch(theme=CREAM_THEME, css=RESPONSIVE_CSS)
