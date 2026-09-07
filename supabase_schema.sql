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

create table if not exists files (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references groups(id) on delete cascade,
    uploader_name text not null,
    file_name text not null,
    storage_path text not null,
    editable boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (group_id, file_name)
);

create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references groups(id) on delete cascade,
    sender_name text not null,
    recipient_name text,
    body text not null,
    created_at timestamptz not null default now()
);

create table if not exists help_requests (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references groups(id) on delete cascade,
    requester_name text not null,
    note text not null,
    status text not null default 'open',
    helper_name text,
    created_at timestamptz not null default now(),
    resolved_at timestamptz
);

create index if not exists idx_members_group_id on members(group_id);
create index if not exists idx_tasks_group_id on tasks(group_id);
create index if not exists idx_tasks_assigned_to on tasks(assigned_to);
create index if not exists idx_files_group_id on files(group_id);
create index if not exists idx_messages_group_id on messages(group_id);
create index if not exists idx_help_requests_group_id on help_requests(group_id);

-- Demo-stage RLS: open policies so the Gradio app (using the anon/service key)
-- can read and write freely. Tighten before any real deployment beyond the
-- challenge demo (see README "Security notes").
alter table groups enable row level security;
alter table members enable row level security;
alter table tasks enable row level security;
alter table files enable row level security;
alter table messages enable row level security;
alter table help_requests enable row level security;

create policy "groups_all" on groups for all using (true) with check (true);
create policy "members_all" on members for all using (true) with check (true);
create policy "tasks_all" on tasks for all using (true) with check (true);
create policy "files_all" on files for all using (true) with check (true);
create policy "messages_all" on messages for all using (true) with check (true);
create policy "help_requests_all" on help_requests for all using (true) with check (true);

-- Storage bucket for uploaded work files. Public so download links work
-- without extra signing logic; same open-access tradeoff as the table
-- policies above (see README "Security notes").
insert into storage.buckets (id, name, public)
values ('group-files', 'group-files', true)
on conflict (id) do nothing;

create policy "group_files_select" on storage.objects
    for select using (bucket_id = 'group-files');
create policy "group_files_insert" on storage.objects
    for insert with check (bucket_id = 'group-files');
create policy "group_files_update" on storage.objects
    for update using (bucket_id = 'group-files');
create policy "group_files_delete" on storage.objects
    for delete using (bucket_id = 'group-files');
