-- ============================================================
-- Research Assistant — Supabase schema
-- Run this entire file once in Supabase SQL Editor.
-- Matches the data-layer design: users, research_runs, documents,
-- threads, all scoped by Row Level Security to auth.uid().
-- ============================================================

-- ---- users ----
-- Supabase already has auth.users (managed by Supabase Auth).
-- This table holds app-specific fields we need alongside it.
create table if not exists public.users (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  tokens_used integer not null default 0,
  quota_limit integer not null default 200000,
  plan text not null default 'free',
  created_at timestamptz not null default now(),
  last_active timestamptz not null default now()
);

alter table public.users enable row level security;

create policy "users_select_own"
  on public.users for select
  using (id = auth.uid());

create policy "users_update_own"
  on public.users for update
  using (id = auth.uid());

-- ---- research_runs ----
create table if not exists public.research_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  query text not null,
  query_hash text,
  status text not null default 'running',
  sub_questions jsonb,
  cycle_count integer not null default 0,
  answer text,
  citations jsonb,
  tokens_used integer default 0,
  estimated_cost_usd numeric(10, 6) default 0,
  mlflow_run_id text,
  created_at timestamptz not null default now()
);

alter table public.research_runs enable row level security;

create policy "research_runs_select_own"
  on public.research_runs for select
  using (user_id = auth.uid());

create policy "research_runs_insert_own"
  on public.research_runs for insert
  with check (user_id = auth.uid());

create policy "research_runs_update_own"
  on public.research_runs for update
  using (user_id = auth.uid());

-- Index for the query cache lookup (query_hash + recency check)
create index if not exists idx_research_runs_query_hash
  on public.research_runs (query_hash, created_at desc);

create index if not exists idx_research_runs_user_id
  on public.research_runs (user_id, created_at desc);

-- ---- documents ----
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  source_url text,
  title text,
  embed_model text not null default 'text-embedding-3-small',
  chunk_count integer default 0,
  status text not null default 'processing',
  ingest_ts timestamptz not null default now()
);

alter table public.documents enable row level security;

create policy "documents_select_own"
  on public.documents for select
  using (user_id = auth.uid());

create policy "documents_insert_own"
  on public.documents for insert
  with check (user_id = auth.uid());

create policy "documents_update_own"
  on public.documents for update
  using (user_id = auth.uid());

-- ---- threads ----
create table if not exists public.threads (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  run_id uuid references public.research_runs(id) on delete cascade,
  title text,
  saved_at timestamptz not null default now()
);

alter table public.threads enable row level security;

create policy "threads_select_own"
  on public.threads for select
  using (user_id = auth.uid());

create policy "threads_insert_own"
  on public.threads for insert
  with check (user_id = auth.uid());

create policy "threads_delete_own"
  on public.threads for delete
  using (user_id = auth.uid());

-- ============================================================
-- Verify: run this after the above to confirm all 4 tables and
-- their RLS policies exist.
-- ============================================================
-- select tablename, rowsecurity from pg_tables
--   where schemaname = 'public';
