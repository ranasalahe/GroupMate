# GroupMate — Explainer Video Script

Target length: ~4-5 minutes. This is the "how it's built" video (backend,
architecture, LLM integration) — different from the Day 8 demo video, which
just shows the app being used.

Record your screen with the app open at `http://localhost:7860` and `app.py`
open in an editor for the code sections. Read the narration in your own
words — this is a script to work from, not to read verbatim.

(Running locally rather than on a deployed Hugging Face Space — Spaces now
requires a paid PRO plan to host Gradio apps, confirmed directly on the
account's own Space-creation page. Documented in the workbook's Day 7
section rather than silently skipped.)

---

## 1. Intro (15s)

**[Screen: GroupMate home page / README on GitHub]**

> "Hi, I'm Rana, and this is GroupMate — an AI-powered app I built for the
> Building AI Application Challenge that fixes the most painful part of
> student group projects: figuring out who does what, fairly."

## 2. The problem (20s)

**[Screen: README or just talk over the app]**

> "Group projects usually split tasks by whoever speaks up first, not by
> who's actually good at what — and nobody notices someone's falling behind
> until it's too late. GroupMate fixes both: AI splits tasks by skill and
> workload, and a dashboard flags anyone trailing the group's pace."

## 3. Architecture overview (40s)

**[Screen: architecture diagram from README, or just narrate over app.py]**

> "The stack is simple on purpose: Python and Gradio for the interface,
> Supabase for the database, and OpenAI's gpt-4o for the intelligence. No
> heavy framework — one Python file, `app.py`, runs the whole thing."

**[Screen: scroll through app.py's table of contents / section comments]**

> "The backend is organized around three AI integration points, plus a
> Supabase schema with six tables: groups, members, tasks, files, messages,
> and help requests."

## 4. The AI — how it actually works (60-90s)

**[Screen: `generate_tags()` function in app.py]**

> "When someone creates a group, the app sends the project description to
> gpt-4o in JSON mode and gets back a set of project-specific skill tags —
> not generic ones like 'teamwork', but things like 'Stripe integration' or
> 'survey design', tailored to that exact project."

**[Screen: `ai_distribute_tasks()` function]**

> "Once every member has picked their strong suits, a second call breaks the
> project into tasks and assigns them by skill match and workload balance —
> again in JSON mode, so the output is always structured data the app can
> act on directly, not free text I'd have to parse."

**[Screen: `_clamp_task_hours()` and the try/except wrapper]**

> "I don't trust the model's output blindly, though. Every AI-reported hour
> estimate gets clamped to a sane range before it's written to the database
> — so a hallucinated number can't silently break the fairness math. And
> every AI or database call is wrapped in error handling, so a flaky
> request shows a friendly message instead of crashing the app."

**[Screen: `ask_ai()` and `_search_file_chunks()`]**

> "The third AI feature is an in-app assistant. It can answer questions
> about the project, and — this is the part I'm most proud of — it can
> search files the team has uploaded. When someone uploads a report, the
> app chunks it, embeds it with OpenAI's embedding model, and stores those
> embeddings in Supabase. When you ask a question, it finds the most
> relevant chunks by cosine similarity and feeds them to the model — so the
> assistant can actually answer questions grounded in your team's own
> documents, with the source file named."

**[Screen: the `BEGIN UNTRUSTED FILE EXCERPTS` fencing in the prompt]**

> "Since that file content comes from other users, I treat it as untrusted
> data in the prompt — clearly fenced off and labeled so the model doesn't
> follow instructions hidden inside an uploaded file. I actually tested
> that live with a direct prompt-injection attempt, and the assistant
> correctly ignored it."

## 5. Live walkthrough (90s)

**[Screen: the running app]**

> "Let me show it end to end."

- Create a group → show the AI-generated tags appearing
- Add a member, submit strong suits → show the automatic task distribution firing once everyone's in
- Dashboard → show the progress rings and the falling-behind flag
- Files tab → upload a file, then ask the AI assistant a question about it, show the cited answer
- Group Chat / Request Help → quick glance

## 6. Evaluation (20s)

**[Screen: GroupMate_Evaluation_Results.pdf]**

> "I didn't just eyeball this — I ran a 12-sample benchmark against real
> project descriptions of different types and group sizes. Every sample
> produced valid, correctly-matched output. Fairness was tight for simple
> projects and wider for larger, role-heavy ones — which told me the model
> leans on skill-fit more than strict hour-balancing, something I've
> documented as a next tuning step."

## 7. Closing (15s)

> "That's GroupMate — built with Gradio, Supabase, and the OpenAI API, with
> real guardrails around the AI output, not just a thin wrapper around a
> chat call. Thanks for watching."

---

## Quick recording tips

- Record in short segments per section — easier to re-take one part than the whole thing.
- Keep code screens on-screen for at least 5-8 seconds so viewers can actually read them.
- Mention briefly that it's running locally (not on a deployed Space) because of the Hugging Face PRO requirement — a one-sentence, matter-of-fact note, not an apology. The checkpoint itself says a video explaining what you've done is fine even if something isn't finished.
