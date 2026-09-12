# GroupMate

An AI-powered task distribution app that fixes the most painful part of student group projects: figuring out who does what fairly.

Members submit their strong suits, and the AI automatically splits project tasks based on skill fit and workload balance, then tracks progress, flags anyone falling behind, and keeps the whole group aligned in one place — chat, file sharing, and help requests included.

Built for the **Building AI Application Challenge 2026** (LLM/API Integration path).

**Live demo:** https://groupmate-glp1.onrender.com
*(free-tier instance — the first request after a period of inactivity can take up to ~50s to wake up)*

## How it works

1. **Create a group** — the admin enters their name, a group name, teammates (comma-separated), a project description, and a deadline. The app calls the OpenAI API (`gpt-4o`) to generate 6-10 project-specific skill tags from the description, and produces a shareable join link + QR code.
2. **Join a group** — teammates paste the join link (or scan the QR code) and enter their name — no separate invite step.
3. **Strong suits** — each member picks their strong suits from the generated tag list. Once everyone has submitted, the app calls OpenAI again to break the project into tasks and assign them fairly by skill-tag match and balanced workload; large tasks can be marked shared across multiple members. AI-reported hour estimates are clamped to a sane range so a hallucinated value can't skew the fairness math.
4. **Dashboard** — a persistent sidebar with:
   - **Home** — personal progress ring, deadline countdown, calendar/schedule view, task checklist, and private 1:1 chat with teammates.
   - **Group Progress** — every member's completion ring, plus an automatic "falling behind" flag (the *nudge*) for anyone trailing the group's time-elapsed pace by more than 20%.
   - **Group Chat** — one shared live feed for the whole group.
   - **File Upload** — shared file library; uploaded text/PDF files are chunked, embedded (`text-embedding-3-small`), and made searchable.
   - **Request Help / Offer Help** — a lightweight way to flag being stuck and for teammates to step in.
   - **Settings** — light/dark mode, admin-approved member removal requests, and a confirmed "Leave Group."
5. **Ask AI** — an in-app assistant available from Home, Group Progress, Group Chat, and File Upload. It retrieves the most relevant chunks from the group's uploaded files by cosine similarity and cites the source file — answers grounded in the team's own documents, not just the model's general knowledge. Retrieved file content is explicitly fenced in the prompt as untrusted data, so text embedded in an uploaded file can't redirect the assistant's behavior (tested live against a real prompt-injection attempt).

## Tech stack

- **UI:** [Gradio](https://gradio.app) — one Python file (`app.py`)
- **Database & storage:** [Supabase](https://supabase.com) (Postgres) — `groups`, `members`, `tasks`, `files`, `messages`, `help_requests`, `file_chunks`, `member_removal_requests` tables, plus a `group-files` storage bucket
- **AI:** OpenAI API — `gpt-4o` (JSON response mode) for tag generation, task distribution, and the Ask AI assistant; `text-embedding-3-small` for the file-search RAG pipeline
- **Deployment:** [Render.com](https://render.com) (free Web Service tier)

## Running locally

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate   # Windows Git Bash
   pip install -r requirements.txt
   ```

2. Set up Supabase:
   - Create a project at [supabase.com](https://supabase.com).
   - Run [`supabase_schema.sql`](supabase_schema.sql) in the Supabase SQL editor to create all tables and the `group-files` storage bucket.

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

   Gradio will print a local URL to open in your browser (defaults to `localhost:7860`; set the `PORT` env var to change it).

## Project structure

```
app.py                  # Gradio UI + Supabase/OpenAI logic (the whole app)
requirements.txt        # Pinned dependencies
supabase_schema.sql     # Database schema (run once in Supabase)
.env.example            # Required environment variables (copy to .env)
scripts/run_eval.py     # Evaluation harness (12-sample benchmark)
docs/                   # Challenge progress workbook, evaluation results, demo videos
```

## Evaluation

A 12-sample benchmark (`scripts/run_eval.py`) runs real project descriptions of varying size/type through the live OpenAI API and checks tag validity and task-assignment name-matching. Results, methodology, and a fairness analysis: [`docs/GroupMate_Evaluation_Results.pdf`](docs/GroupMate_Evaluation_Results.pdf).

## Security notes

- API keys are read from environment variables only — never hardcoded.
- Uploaded-file content used by Ask AI is fenced as untrusted data in the prompt, so it can't be used to inject instructions into the assistant.
- The Supabase schema ships with open row-level-security policies for demo simplicity. This is a known limitation, tracked as a post-challenge hardening item (see `docs/progress-workbook.md`).

## Roadmap

Beyond the challenge scope: personalized AI-generated weekly schedules per member, and richer conflict-resolution guidance from the AI assistant.
