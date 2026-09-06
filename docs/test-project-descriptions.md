# GroupMate Evaluation Set

12 sample project descriptions used to test tag generation and task
distribution across a range of project types, sizes, and group sizes.
For each one, run `generate_tags()` then `ai_distribute_tasks()` with the
listed member roster and check against the criteria below.

## Evaluation criteria

For every sample, check:

1. **Tag validity** — `generate_tags` returns valid JSON with 6-10 tags,
   all specific to the project (no generic soft skills like "teamwork").
2. **Task validity** — `ai_distribute_tasks` returns valid JSON matching
   the schema (`title`, `estimated_hours`, `assigned_to`, `shared`,
   `shared_with`), and every `assigned_to`/`shared_with` name matches a
   real member name exactly.
3. **Fairness** — total `estimated_hours` per member should be roughly
   balanced (no member with more than ~1.5x another's load, absent a
   skill-based reason).
4. **Shared-task logic** — at least one large/broad task should be
   flagged `shared` when the group is small relative to project scope,
   and no task should be marked shared unnecessarily for a straightforward
   solo-sized task.

## Samples

| # | Project description | Members |
|---|---|---|
| 1 | Build a mobile app that helps students split shared apartment bills, with push notifications for due payments. | Ali (frontend, UI design), Sara (backend, databases), Omar (no strong suits selected) |
| 2 | Research report on the economic impact of remote work on the UAE real estate market, including survey data collection and analysis. | Layla (survey design, statistics), Yousef (writing, research) |
| 3 | Design and pitch a marketing campaign for a fictional sustainable fashion brand, including a brand deck and social media plan. | Mona (graphic design), Fahad (marketing strategy), Nour (copywriting), Zaid (no strong suits selected) |
| 4 | Build a full-stack e-commerce website with product listings, a shopping cart, and Stripe payment integration. | Hassan (React frontend), Reem (Node.js backend), Tariq (payment integration, security), Dana (QA testing) |
| 5 | Create a short documentary film about student mental health on campus, including interviews and editing. | Huda (video editing), Karim (interviewing, scriptwriting) |
| 6 | Develop a machine learning model to predict student grades from study habits, with a written report on findings. | Salma (data science, Python), Adel (statistics), Rania (report writing), Bilal (no strong suits selected), Nabil (no strong suits selected) |
| 7 | Plan and execute a charity fundraising event for a local shelter, including logistics, sponsorship outreach, and a budget. | Yasmin (event planning), Khaled (sponsorship/outreach), Lina (budgeting) |
| 8 | Build a Discord bot that manages study group scheduling and sends reminder messages. | Marwan (Python, bot development) |
| 9 | Conduct a UX research study comparing two competing food delivery apps, culminating in a usability report with recommendations. | Farah (UX research), Ibrahim (data analysis), Noor (report writing) |
| 10 | Design a board game about climate change for high school classrooms, including rules, prototype art, and a teacher's guide. | Aisha (game design), Waleed (illustration), Huda (curriculum writing), Sami (no strong suits selected), Talal (no strong suits selected), Rasha (no strong suits selected) |
| 11 | Build an internal tool that scrapes and aggregates internship postings from five university career sites into one dashboard. | Ahmad (web scraping, Python), Dalia (frontend dashboard) |
| 12 | Write and record a 6-episode podcast series on entrepreneurship in the Gulf, including guest outreach and editing. | Jana (audio editing), Sultan (guest outreach, interviewing), Maya (research, scriptwriting), Fadi (no strong suits selected) |

## Results log

Fill in after each run once real API keys are configured (see README).

| # | Tags valid? | Tasks valid? | Fair split? | Shared-task logic correct? | Notes |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |
| 11 | | | | | |
| 12 | | | | | |
