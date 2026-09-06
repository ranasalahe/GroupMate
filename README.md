# GroupMate

An AI-powered task distribution app that fixes the most painful part of student group projects: figuring out who does what fairly.

Members submit their strong suits, and the AI automatically splits project tasks based on skill fit and workload balance, then tracks progress, flags anyone falling behind, and keeps the whole group aligned in one place.

Built for the **Building AI Application Challenge 2026** (LLM/API Integration path).

## How it works

1. **Create a group** — enter the project name, description, and deadline. The app calls the OpenAI API to generate 6-10 project-specific skill tags from the description.
2. **Add members** — add each teammate by name.
3. **Strong suits** — each member picks their strong suits from the generated tag list.
4. **Task distribution** — once every member has submitted, the app calls the OpenAI API again to break the project into tasks and assign them fairly, based on skill-tag match and balanced workload. Large tasks can be marked as shared across multiple members.
5. **Dashboard** — live per-member and group progress, a deadline countdown, and a falling-behind flag for anyone trailing the group's pace.
6. **My Tasks** — each member checks off their own tasks as they complete them.

## Tech stack

- **UI:** [Gradio](https://gradio.app)
- **Backend logic:** Python
- **Database:** [Supabase](https://supabase.com) (Postgres) — `groups`, `members`, `tasks` tables
- **AI:** OpenAI API (`gpt-4o`), JSON response mode, for tag generation and task distribution
- **Deployment:** Hugging Face Spaces

## Running locally

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate   # Windows Git Bash
   pip install -r requirements.txt
   ```

2. Set up Supabase:
   - Create a project at [supabase.com](https://supabase.com).
   - Run [`supabase_schema.sql`](supabase_schema.sql) in the Supabase SQL editor to create the `groups`, `members`, and `tasks` tables.

3. Copy `.env.example` to `.env` and fill in your keys:

   ```
   OPENAI_API_KEY=sk-...
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-key
   ```

4. Run the app:

   ```bash
   python app.py
   ```

   Gradio will print a local URL to open in your browser.

## Project structure

```
app.py               # Gradio UI + Supabase/OpenAI logic (the whole app)
requirements.txt      # Pinned dependencies
supabase_schema.sql   # Database schema (run once in Supabase)
.env.example           # Required environment variables (copy to .env)
docs/progress-workbook.md   # Challenge day-by-day progress log
```

## Security notes

- API keys are read from environment variables only — never hardcoded.
- The Supabase schema ships with open row-level-security policies for demo simplicity. This is a known limitation, tracked as a post-challenge hardening item (see `docs/progress-workbook.md`, Day 6).

## Roadmap

Beyond the challenge MVP: group chat, a "help a groupmate" flow, personalized AI weekly schedules per member, and a private AI chat for guidance and conflict resolution.
