"""Run the evaluation set (docs/test-project-descriptions.md) against the
live OpenAI calls and print a fairness/validity report.

Usage:
    python scripts/run_eval.py

Requires OPENAI_API_KEY in the environment or .env. Does not touch
Supabase — it calls generate_tags() and ai_distribute_tasks() directly.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ai_distribute_tasks, generate_tags  # noqa: E402

SAMPLES = [
    (
        "Build a mobile app that helps students split shared apartment bills, "
        "with push notifications for due payments.",
        ["Ali (frontend, UI design)", "Sara (backend, databases)", "Omar"],
    ),
    (
        "Research report on the economic impact of remote work on the UAE "
        "real estate market, including survey data collection and analysis.",
        ["Layla (survey design, statistics)", "Yousef (writing, research)"],
    ),
    (
        "Design and pitch a marketing campaign for a fictional sustainable "
        "fashion brand, including a brand deck and social media plan.",
        [
            "Mona (graphic design)",
            "Fahad (marketing strategy)",
            "Nour (copywriting)",
            "Zaid",
        ],
    ),
    (
        "Build a full-stack e-commerce website with product listings, a "
        "shopping cart, and Stripe payment integration.",
        [
            "Hassan (React frontend)",
            "Reem (Node.js backend)",
            "Tariq (payment integration, security)",
            "Dana (QA testing)",
        ],
    ),
    (
        "Create a short documentary film about student mental health on "
        "campus, including interviews and editing.",
        ["Huda (video editing)", "Karim (interviewing, scriptwriting)"],
    ),
    (
        "Develop a machine learning model to predict student grades from "
        "study habits, with a written report on findings.",
        [
            "Salma (data science, Python)",
            "Adel (statistics)",
            "Rania (report writing)",
            "Bilal",
            "Nabil",
        ],
    ),
    (
        "Plan and execute a charity fundraising event for a local shelter, "
        "including logistics, sponsorship outreach, and a budget.",
        ["Yasmin (event planning)", "Khaled (sponsorship/outreach)", "Lina (budgeting)"],
    ),
    (
        "Build a Discord bot that manages study group scheduling and sends "
        "reminder messages.",
        ["Marwan (Python, bot development)"],
    ),
    (
        "Conduct a UX research study comparing two competing food delivery "
        "apps, culminating in a usability report with recommendations.",
        ["Farah (UX research)", "Ibrahim (data analysis)", "Noor (report writing)"],
    ),
    (
        "Design a board game about climate change for high school classrooms, "
        "including rules, prototype art, and a teacher's guide.",
        [
            "Aisha (game design)",
            "Waleed (illustration)",
            "Huda (curriculum writing)",
            "Sami",
            "Talal",
            "Rasha",
        ],
    ),
    (
        "Build an internal tool that scrapes and aggregates internship "
        "postings from five university career sites into one dashboard.",
        ["Ahmad (web scraping, Python)", "Dalia (frontend dashboard)"],
    ),
    (
        "Write and record a 6-episode podcast series on entrepreneurship in "
        "the Gulf, including guest outreach and editing.",
        [
            "Jana (audio editing)",
            "Sultan (guest outreach, interviewing)",
            "Maya (research, scriptwriting)",
            "Fadi",
        ],
    ),
]


def parse_member(raw: str) -> dict:
    if "(" in raw:
        name, tags = raw.split("(", 1)
        return {"name": name.strip(), "strong_suits": [t.strip() for t in tags.rstrip(")").split(",")]}
    return {"name": raw.strip(), "strong_suits": []}


def main():
    for i, (description, raw_members) in enumerate(SAMPLES, start=1):
        members = [parse_member(m) for m in raw_members]
        print(f"\n=== Sample {i} ===")
        print(description[:80] + ("..." if len(description) > 80 else ""))

        try:
            tags = generate_tags(description)
            tags_ok = 6 <= len(tags) <= 10
            print(f"Tags ({len(tags)}, valid={tags_ok}): {tags}")
        except Exception as exc:
            print(f"TAG GENERATION FAILED: {exc}")
            continue

        try:
            tasks = ai_distribute_tasks(description, members)
        except Exception as exc:
            print(f"TASK DISTRIBUTION FAILED: {exc}")
            continue

        names = {m["name"] for m in members}
        hours_per_member = {m["name"]: 0.0 for m in members}
        bad_names = []
        for t in tasks:
            owners = [t.get("assigned_to")] + list(t.get("shared_with", []))
            for owner in owners:
                if owner in hours_per_member:
                    hours_per_member[owner] += t.get("estimated_hours", 0) / len(owners)
                elif owner:
                    bad_names.append(owner)

        print(f"Tasks: {len(tasks)}")
        print(f"Hours per member: { {k: round(v, 1) for k, v in hours_per_member.items()} }")
        if len(hours_per_member) > 1:
            spread = max(hours_per_member.values()) - min(hours_per_member.values())
            print(f"Hour spread: {round(spread, 1)} (stdev {round(statistics.pstdev(hours_per_member.values()), 2)})")
        if bad_names:
            print(f"WARNING — names not matching any member: {bad_names}")


if __name__ == "__main__":
    main()
