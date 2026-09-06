-- GroupMate Supabase schema
-- Run this in the Supabase SQL editor for your project.

create extension if not exists "pgcrypto";

create table if not exists groups (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    description text not null,
    deadline timestamptz not null,
    tags jsonb not null default '[]',
    created_at timestamptz not null default now()
);

create table if not exists members (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references groups(id) on delete cascade,
    name text not null,
    strong_suits jsonb not null default '[]',
    submitted boolean not null default false,
    created_at timestamptz not null default now()
);

create table if not exists tasks (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references groups(id) on delete cascade,
    title text not null,
    estimated_hours numeric not null default 1,
    assigned_to uuid references members(id) on delete set null,
    shared boolean not null default false,
    shared_with jsonb not null default '[]',
    completed boolean not null default false,
    created_at timestamptz not null default now()
);

create index if not exists idx_members_group_id on members(group_id);
create index if not exists idx_tasks_group_id on tasks(group_id);
create index if not exists idx_tasks_assigned_to on tasks(assigned_to);

-- Demo-stage RLS: open policies so the Gradio app (using the anon/service key)
-- can read and write freely. Tighten before any real deployment beyond the
-- challenge demo (see README "Security notes").
alter table groups enable row level security;
alter table members enable row level security;
alter table tasks enable row level security;

create policy "groups_all" on groups for all using (true) with check (true);
create policy "members_all" on members for all using (true) with check (true);
create policy "tasks_all" on tasks for all using (true) with check (true);
