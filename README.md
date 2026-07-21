# Research Assistant — Core Agent Graph

This is the working LangGraph implementation of the agent loop we
designed: `quota_check → decomposer → router → (vector | web | both)
→ grader → synthesizer → reflection → [loop back to router OR formatter]`.

## Files

- **`state.py`** — `ResearchState` TypedDict. Pay attention to which
  fields use `Annotated[list, operator.add]` (append across cycles)
  vs plain types (overwrite each time). Getting this wrong is the
  most common LangGraph bug.
- **`nodes.py`** — every node function. External calls (OpenAI,
  Pinecone, Tavily) are stubbed in small `_call_*` functions marked
  `REAL:` — swap their bodies for actual API calls, the node
  functions themselves don't need to change.
- **`build.py`** — wires nodes into the `StateGraph`, including the
  conditional edges (`check_quota`, `check_decomposition`,
  `pick_sources`, `route_after_reflection`).
- **`run_demo.py`** — runnable smoke test. Streams every node's
  state delta to stdout, then re-runs to print the true final
  reduced state (proves the reducers work).
- **`test_rejection_path.py`** — proves edge case 1.1 (topic-less
  query) short-circuits to `error_handler` before any retrieval or
  synthesis call happens — protecting against wasted cost and
  hallucinated answers on empty input.

## Run it

```bash
pip install langgraph langchain-core
cd backend
python -m app.graph.run_demo            # full agent loop, forces 3 reflection cycles
python -m app.graph.test_rejection_path  # proves the reject-before-retrieval guard
```

## LLM integration (real API calls)

`app/llm/client.py` and `app/llm/prompts.py` wire real OpenAI calls
into every node. Two functions carry all the weight:

- **`call_json()`** — for structured output (decomposer, grader,
  reflection). Retries 3x with exponential backoff on transient
  errors (`RateLimitError`, `APITimeoutError`, `APIError`) via
  `tenacity`, then attempts a JSON repair pass (strips markdown
  fences, fixes trailing commas) before raising `LLMCallError`.
- **`call_text()`** — for free-form prose (synthesizer only). Same
  retry policy, no JSON parsing.

Both are the single seam every node calls through — no node touches
the OpenAI SDK directly, so retry policy and model selection live in
exactly one place.

**Model selection per node** (from the cost modeling design):
`gpt-4o-mini` for decomposition/grading/reflection (classification-
style tasks), `gpt-4o` only for synthesis (long-form reasoning where
quality justifies the ~25x cost delta).

**Citation validation** (`_extract_and_validate_citations` in
`nodes.py`) runs after every synthesis call — strips any `[N]` the
LLM hallucinates that doesn't correspond to a real graded chunk
(edge case 4.1). Tested end-to-end in
`tests/test_llm_call_integration.py`, not just in isolation.

⚠️ **No live API key in this environment** — the code is real and
fully wired, but was never called against the actual OpenAI API here.
Tests mock at the OpenAI client boundary (`get_client()`), proving
prompt construction, retry logic, and JSON repair all work correctly
against realistic response shapes. Set `OPENAI_API_KEY` and run
`python -m app.graph.run_demo` to see it hit the real API.

## Retrieval integration (Pinecone + Tavily, real API calls)

`app/retrieval/base.py` defines a `VectorStore` ABC — the abstraction
we specifically scrutinized earlier as the highest-value, lowest-effort
fix against Pinecone vendor lock-in. **No node imports Pinecone
directly** — only `app/retrieval/pinecone_store.py` does. Swapping to
Qdrant later means writing one new `VectorStore` implementation, not
touching every call site.

- **`pinecone_store.py`** — `PineconeVectorStore`, using the real
  `AsyncPinecone` + `AsyncIndex` client so retrieval never blocks the
  FastAPI event loop. Lazily constructs its client on first use (same
  pattern as `app.llm.client.get_client()`), so importing the module
  never requires `PINECONE_API_KEY` to be set.
- **`tavily_search.py`** — `search_web()`, always calling
  `search_depth="advanced"` (never Tavily's default) per our earlier
  design: basic depth returns ~200-char snippets, advanced returns
  full extracted article text the synthesizer actually needs.
- **`app/llm/embeddings.py`** — `embed_text()`/`embed_texts()`,
  wrapping OpenAI's embeddings endpoint with the same retry policy as
  the chat completion client.

`_call_vector_retriever` and `_call_web_search` in `nodes.py` fan out
one embedding + Pinecone query (or one Tavily search) **per
sub-question concurrently** via `asyncio.gather` — proven by timing
tests in `test_retriever_orchestration.py`, not just inferred from
the code.

⚠️ **No live Pinecone or Tavily keys in this environment either** —
same situation as the OpenAI integration. All code is real and fully
wired; tests mock at the client boundary (`AsyncIndex.query/upsert`,
`AsyncTavilyClient.search`) using accurately-shaped fake response
objects (`QueryResponse`, `ScoredVector`), proving our wrapper logic
handles realistic responses correctly. Set `PINECONE_API_KEY`,
`PINECONE_INDEX_HOST`, and `TAVILY_API_KEY` to hit the real APIs.

## Ingestion pipeline

`app/ingestion/` is the offline pipeline from the ingestion design
session: `source → loader → chunker → metadata tagger → embedder →
upsert`. It exists independently of the query-time graph — nothing
currently calls it from an HTTP route (see "Next integration steps"),
but the full pipeline is built, wired, and tested.

- **`loaders.py`** — normalizes PDF bytes, a URL, or raw text into
  `{text, title}`. Enforces the 20MB PDF size cap (DoS guard) *before*
  any parsing is attempted, rejects password-protected PDFs with a
  clear error instead of crashing, and survives a single malformed
  page without losing the rest of the document. `load_url` strips
  `<script>`/`<style>`/`<nav>`/`<footer>` blocks entirely (not just
  their tags) before extracting text.
- **`chunker.py`** — 512-token chunks with 64-token overlap, recursive
  paragraph→sentence→word splitting that never cuts mid-word, an
  orphan-chunk filter (drops anything under 80 tokens — edge case
  2.4), and figure/table reference stripping (replaces `"as shown in
  Figure 3"` with `"[visual reference omitted]"` — silent failure S4,
  since PDF text extraction loses the actual figures).
- **`orchestrator.py`** — ties it together via `ingest_source()`.
  Vector IDs are `sha256(source_identifier :: chunk_index)`, making
  re-ingestion **idempotent** — the same document overwrites its own
  vectors instead of duplicating them. Every chunk's metadata is
  stamped with `embed_model` (needed for the cache-key design and
  catching embedding-model drift). Depends only on the `VectorStore`
  interface, never on Pinecone directly.

### A real bug this pipeline surfaced: `tiktoken`'s network dependency

`chunker.py` originally called `tiktoken.get_encoding("cl100k_base")`
at **module import time**. That function downloads its vocabulary
file over the network on first use — and in this sandbox, that
download domain isn't in the egress allowlist, so simply *importing*
the chunker crashed the entire test suite with a `403 Forbidden`.

This isn't just a sandbox quirk — the same failure mode hits real
deployments too: slower cold starts on every fresh container, total
unimportability in network-restricted CI runners or corporate
networks, and a transient network blip during deploy taking down
ingestion entirely. Fixed by lazy-loading the encoding on first real
use, caching the result (success *or* failure, so a failing network
call is never retried per-chunk), and falling back to an approximate
character-based token counter (~4 chars/token) if the download
genuinely fails — the same "external dependency degrades, never
crashes" pattern used for Pinecone (`degraded=True`) and MLflow
(observability failures never break the critical path) elsewhere in
this project. `TestTiktokenUnavailableFallback` in `test_chunker.py`
permanently regression-tests this path, including a specific
assertion that a failed download is never retried more than once.

⚠️ **No live PDF-loading, URL-fetching, embedding, or Pinecone-upsert
calls have been made against real infrastructure** — same caveat as
the rest of this project. `httpx` calls in `load_url` are mocked via
`httpx.MockTransport`; PDFs are generated with `reportlab` to get
genuine parseable PDF structure without any external file.

## ⚠️ Auth security fix — read this first

An earlier version of this project shipped `app/api/deps.py` with
`verify_jwt()` as a documented stub: **any non-empty string was
accepted as a valid bearer token.** It was clearly labeled as a stub
in the code, but a labeled hole is still a hole — a person auditing
this project correctly called that out as the single most serious
thing standing between "backend with good test coverage" and
"backend that's actually complete."

This is now fixed with **real cryptographic verification**:

- **`app/api/deps.py`** — `verify_jwt()` inspects the token's own
  `alg` header and verifies accordingly, since Supabase issues JWTs
  under one of two schemes depending on project age/settings:
  - **HS256** (legacy) — verified against a shared secret,
    `SUPABASE_JWT_SECRET`.
  - **ES256** (current default for all new Supabase projects since
    October 2025) — verified against the project's public JWKS,
    fetched from `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`.
  Audience (`aud="authenticated"`), expiry, and the `sub` claim are
  all checked. A tampered signature, wrong secret, expired token,
  wrong audience, malformed token, or unsupported algorithm (including
  the classic `alg: none` unsigned-token attack) are all rejected.
- **`app/api/jwks.py`** — async JWKS fetching with a 600-second cache
  (matching Supabase's own documented edge-cache TTL), with a
  once-only cache-bypass refetch if a token's `kid` isn't found — the
  expected behavior right after a legitimate key rotation, not an
  error condition by itself.
- **Deliberately no silent fallback.** If neither `SUPABASE_JWT_SECRET`
  nor `SUPABASE_URL` is configured, this fails loudly — `AuthConfigError`
  → HTTP 500 — rather than silently accepting anything, which is
  exactly what the previous version did. A misconfigured deployment
  should be *obviously* broken, not *quietly* insecure.

`tests/api/test_deps.py` proves all of this directly: a genuinely
valid token is accepted; a token signed with the wrong secret,
expired, wrong audience, missing `sub`, unsigned (`alg: none`), or
just an arbitrary non-JWT string are all rejected; and the full
asymmetric path is proven with a **real generated EC keypair** — a
token is signed with a real private key, the public key is served as
a mocked JWKS document, and verification succeeds using only the
public half, exactly matching how a real Supabase project's ES256
tokens would be verified.

Every test file that previously used arbitrary strings like
`"Bearer valid-token"` (which the old stub accepted, but which real
verification correctly rejects as an invalid signature) now mints a
real, properly signed test JWT via a shared `make_bearer_header()`
helper in `tests/api/conftest.py`. The live HTTP smoke test
(`run_http_smoke_test.py`) does the same, and was re-run against this
fix to confirm the full HTTP + SSE + ingestion flow still works
end-to-end with genuine token verification in the loop, not just unit
tests in isolation.

## ⚠️ Quota / cost control fix

The other gap flagged in the same audit: `quota_check_node` always
returned `quota_ok=True`. It was checked at the right point in the
graph (before any LLM call), but the value it checked never changed —
functionally identical to having no check at all. A user could run
unlimited research queries with zero cost control.

This is now fixed with **real per-user token tracking**, closing the
full check → spend → record loop:

- **`app/quota_store.py`** — `QuotaStore`, an in-memory stand-in for
  the Supabase `users.tokens_used`/`quota_limit` columns (same
  swappable-seam pattern as `RunStore`/`JobStore`). `has_quota()` is
  checked by `quota_check_node` before a run starts; `add_usage()` is
  called after every run completes — successful *or* failed — with
  the real total tokens consumed.
- **`app/llm/client.py`** — `call_json()`/`call_text()` now return
  `(payload, tokens_used)` tuples instead of just the payload,
  extracting the real `total_tokens` from every OpenAI response's
  `usage` field. This is not a cosmetic change — it's the only reason
  the quota system has real numbers to work with instead of always
  recording 0.
- **`app/graph/state.py`** — `tokens_used` is now `Annotated[int,
  operator.add]`, the same reducer pattern as `chunks`, so usage
  correctly SUMS across every LLM call in a run — including every
  chunk graded and every reflection cycle — rather than only
  capturing the last node's contribution.
- **`formatter_node` and `error_handler_node`** both record usage via
  a shared `_record_usage()` helper. Both, deliberately — a query
  rejected by `check_decomposition` (e.g. a topic-less query) still
  made one real, billable decomposer call before being rejected, and
  that spend must still count against quota rather than being
  silently forgotten because the run never reached `formatter_node`.

`tests/test_quota_store.py` and the rewritten `test_guard_and_error_nodes.py`
prove the store and the two recording node paths in isolation.
`tests/test_graph_integration.py` proves the full loop end-to-end
against the **real compiled graph**: a user already over their limit
is blocked before the decomposer ever runs (verified by node
visitation, not just by reading the conditional-edge code), and a
successful run's `tokens_used` both accumulates correctly (620 across
decomposer + 2× grader + synthesizer + reflection, hand-computed and
matched exactly) and gets recorded into `QuotaStore` by the time the
run completes.

## Resumable runs — LangGraph checkpointer

Previously, if the FastAPI process crashed mid-run, that run was
simply gone — LangGraph holds all execution state in memory only, so
a restart starts from nothing.

- **`app/graph/checkpointing.py`** — manages an `AsyncSqliteSaver`'s
  lifecycle explicitly (open at app startup, close at shutdown) rather
  than via its `from_conn_string()` async context manager, since the
  connection needs to live for the entire process, not just one
  request. `init_checkpointer()` degrades non-fatally on failure — an
  unwritable checkpoint path falls back to non-resumable runs, the
  same principle as the Pinecone warmup ping, never crashing startup.
- **`app/graph/build.py`** — `build_research_graph()` now accepts an
  optional `checkpointer` parameter. Every existing test and demo
  script builds/imports the module-level `research_graph` (no
  checkpointer, unchanged behavior); the real FastAPI app gets a
  second, checkpointed graph built once at startup via
  `get_research_graph()`, which transparently falls back to the plain
  graph if checkpointing never initialized.
- **`app/graph/runner.py`** — every `astream()` call now passes
  `config={"configurable": {"thread_id": run_id}}`, tying LangGraph's
  checkpoints to a specific run. A new `resume_graph_and_publish()`
  resumes a run from its last checkpoint (`astream(None, ...)`
  instead of a fresh initial state) rather than restarting it.

`tests/test_checkpointing.py` proves this for real, not just that the
pieces are wired: a graph compiled with `interrupt_after=["grader"]`
is made to pause **right after grading, before synthesis ever runs**
— simulating a process crash at that exact point without needing to
actually kill anything. Resuming from that checkpoint on the same
`thread_id` is then verified two ways: `decomposer`/`router`/`grader`
do **not** run a second time (proven by node visitation), and the
final answer's citations correctly reference the chunks retrieved and
graded *before* the interruption — proving the checkpoint preserved
full state, not just "which node to run next."

**Scope note, stated honestly:** this makes the *graph's own*
execution state resumable within a running process. Full
resumability across a genuine crash-and-restart also needed
`RunStore` (previously in-memory) to survive the restart, so the
FastAPI process still knows a run was in flight and can reconnect a
client's SSE stream to it — that gap is now closed too, see the next
section.

## Supabase-backed stores — `RunStore`, `JobStore`, `QuotaStore`

The three in-memory stores now have real Postgres-backed
counterparts, selected transparently at call time via
`get_run_store()` / `get_job_store()` / `get_quota_store()` — every
route, node, and runner calls these getters instead of importing a
raw singleton, so the switch from in-memory to Supabase-backed
requires zero changes anywhere else in the codebase. `app.db.
is_db_enabled()` (true once `init_pool()` — wired into `main.py`'s
`lifespan`, called with `DATABASE_URL` — succeeds) decides which
backend each getter returns.

- **`app/quota_store.py` — `SupabaseQuotaStore`** — the simplest of
  the three: pure reads/writes against the `users` table, no
  live-streaming concern at all. `add_usage()`/`set_limit()` use
  `INSERT ... ON CONFLICT DO UPDATE` so a first-time user's row is
  created transparently rather than requiring a separate signup-time
  insert to already exist.
- **`app/ingest_store.py` — `SupabaseJobStore`** — backed by the
  `documents` table (see `supabase_schema_additions.sql` for the
  `error`/`chunks_dropped` columns this needed beyond the original
  schema). Also no live-streaming concern — ingestion status is
  *polled*, not streamed, by design.
- **`app/store.py` — `SupabaseRunStore` — a genuine hybrid, not a
  naive swap.** Live SSE fan-out (`publish_event`/`subscribe`/
  `unsubscribe`) is inherently process-local: Postgres has no push
  mechanism the app can `await` on, and the whole point of
  `asyncio.Queue` per subscriber is in-process delivery. A real
  cross-process fan-out needs a message broker (Redis pub/sub — the
  natural fit, per the Docker/deployment design's Celery+Redis
  discussion), which is out of scope here. So: `create_run` writes
  through to *both* an internal in-memory `RunStore` (for live
  pub/sub) and Postgres (for durability); `get_result`/`get_status`/
  `owns` read *only* from Postgres, since those must stay correct
  across a process restart even when the in-memory record is long
  gone. `tests/test_supabase_run_store.py` proves this boundary
  directly — `get_result`/`owns` are tested against a store that
  never had `create_run` called on it at all, simulating exactly the
  post-restart case where no in-memory record could possibly exist.

**Interface-parity refactor this required, stated honestly:** the
in-memory stores' *read* methods (`get_status`, `get_result`, `owns`,
`has_quota`, `get_usage`) were originally synchronous, since a dict
lookup doesn't need to be async — but a genuine Postgres read
absolutely does. For `get_run_store()` etc. to return either backend
interchangeably, every method had to become `async def`, including
the in-memory ones that don't strictly need to await anything. This
was a mechanical but correctness-critical change, touching every
call site in `routes.py`, `sse.py`, both runners, and every existing
test — the same shape of ripple effect as the `call_json()` tuple
contract change during the quota fix, for the same underlying reason:
a shared interface has to be designed for its *most* demanding
implementation, not its simplest one.

All three Supabase-backed stores are tested against a hand-rolled
`FakePool` (`tests/fake_pool.py`) — a scriptable stand-in for
`asyncpg.Pool`'s `fetchrow`/`execute`/`fetch` interface, never a real
database. This proves the actual SQL query logic (row-found vs
unknown-user defaults, the `ON CONFLICT` upsert pattern, the
`documents` table's column-name mapping to `IngestionResult`'s field
names) works correctly, not just that plausible-looking SQL strings
were written. `app/db.py`'s connection-pool lifecycle
(`init_pool`/`close_pool`/`is_db_enabled`) is tested the same way,
mocking `asyncpg.create_pool` — including the non-fatal degrade path
when the DSN is bad or the database is unreachable, matching the same
pattern already established for the checkpointer and Pinecone warmup.

⚠️ **No live Postgres connection has been used anywhere in this
project** — same caveat as OpenAI/Pinecone/Tavily. Every test mocks
at the `asyncpg.Pool` boundary. Run the schema files
(`supabase_schema.sql` then `supabase_schema_additions.sql`) against
a real Supabase project, set `DATABASE_URL`, and this switches on
with zero code changes.

## FastAPI + SSE streaming layer

The full HTTP API sits on top of the graph, matching the API design
session exactly:

- **`app/main.py`** — FastAPI app with a `lifespan` startup hook that
  fires the Pinecone warmup ping (the highest-value/lowest-effort fix
  against serverless cold start from the tool-scrutiny session).
  Warmup failure is caught and logged, never crashes startup.
- **`app/api/deps.py`** — `get_current_user_id()`, the ONLY source of
  `user_id` anywhere in the request lifecycle. Never read from the
  request body — matches the "namespace must come from the verified
  JWT" security rule. See the auth security fix section above — this
  now performs real verification, not a stub.
- **`app/api/schemas.py`** — `QueryRequest` enforces `min_length=10` /
  `max_length=1000` directly implementing the input edge cases we
  designed (trivial/empty queries rejected, pasted essays redirected
  to `/ingest/document` rather than silently truncated).
- **`app/store.py`** — `RunStore`, an in-memory stand-in for the
  Supabase `research_runs` table with the same external shape
  (create/publish/subscribe/get_result), so swapping in real Postgres
  later touches only this file. `owns(run_id, user_id)` is the
  in-memory equivalent of Supabase RLS — every route checks it before
  returning any data, independently blocking cross-user access even
  if something upstream got `user_id` wrong.
- **`app/api/sse.py`** — the SSE generator. Implements disconnect
  detection (edge case 6.2 — polls `request.is_disconnected()` every
  0.5s so a closed browser tab stops the generator instead of pushing
  events into a queue nobody reads) and terminal-event handling
  (`run_complete`/`error` end the stream themselves).
- **`app/graph/runner.py`** — bridges LangGraph's `astream()` to the
  `RunStore`. Uses `stream_mode=["updates", "values"]` in a single
  call — 'updates' become live `node_complete` SSE events, the final
  'values' chunk becomes the true final state. This is the fix for
  the state-reconstruction bug we caught earlier in `run_demo.py`,
  applied without needing to run the graph twice. `_summarize_delta()`
  explicitly allowlists what crosses the wire per node — raw chunk
  text and internal error strings never leak to the client.
- **`app/api/routes.py`** — `POST /research/query` returns a `run_id`
  in ~ms via `BackgroundTasks` (the client opens the SSE connection
  immediately, never waiting synchronously); `GET
  /research/{run_id}/stream`; `GET /research/{run_id}/result`. Every
  run-scoped route checks `run_store.owns()` first and returns **404,
  not 401/403**, for unauthorized access — doesn't leak whether a
  `run_id` exists to someone who doesn't own it.

### Ingestion routes

`POST /ingest/document` and `GET /ingest/{job_id}/status` close the
last gap between the ingestion pipeline (built earlier) and the HTTP
layer — `ingest_source()` existed and was fully tested, but nothing
called it from an HTTP route until now.

- Accepts **exactly one** of: a PDF file upload (`multipart/form-data`,
  field `file`), a `url` form field, or a `text` form field — mapping
  directly to `ingest_source()`'s three `source_type` values. Providing
  zero or more than one is a `422`.
- **Fails fast on oversized uploads** — checks `UploadFile.size` (when
  the transfer encoding provides it) against the same
  `MAX_PDF_SIZE_BYTES` limit `loaders.py` enforces, returning `413`
  before ever buffering the file into memory. `load_pdf()`'s own check
  remains as defense in depth for cases where `.size` isn't available.
- Same `BackgroundTasks` + immediate-response pattern as
  `/research/query`: returns a `job_id` right away, does the actual
  parsing/chunking/embedding/upsert work in the background (a 20-page
  PDF can take several seconds — far too slow to hold a request open).
- **`app/ingest_store.py`** — `JobStore`, deliberately simpler than
  `RunStore`: ingestion status is *polled*, not streamed (per the
  original design), so there's no pub/sub or replay-buffer machinery,
  just `create_job` / `set_result` / `set_error` / `owns`. Same
  swappable-seam and RLS-equivalent `owns()` guarantee as `RunStore`.
- **`app/ingestion/runner.py`** — the background task, with the exact
  same top-level exception-safety principle as `app/graph/runner.py`:
  a `LoaderError` (bad input — oversized file, corrupt PDF, unreachable
  URL) is logged at INFO and marks the job `failed` with a clear
  message; any *other* exception (e.g. a genuine OpenAI outage during
  embedding) is caught too, logged at ERROR, and still marks the job
  `failed` rather than leaving it stuck at `processing` forever.

### Live HTTP smoke test

`app/graph/run_http_smoke_test.py` drives the **real FastAPI app**
(via Starlette's `TestClient`, running the actual ASGI app in-process)
through the full flow a real browser would: unauthenticated request
rejected (401), research run started (202 + `run_id`), cross-user
access blocked (404), full SSE event stream read node-by-node in
order, final result fetched and asserted — **and now the full ingestion
flow too**: a document ingested via `POST /ingest/document` (text
mode), cross-user access to the ingest job blocked (404), and the
final job status polled and asserted (`chunks_ingested > 0`). Only the
three true external SDK boundaries are mocked; every line of HTTP
routing, auth, SSE formatting, ingestion, and graph orchestration runs
for real.

```bash
cd backend
python -m app.graph.run_http_smoke_test
```

## Run the real server

```bash
pip install -r requirements.txt
cd backend
export OPENAI_API_KEY=... PINECONE_API_KEY=... PINECONE_INDEX_HOST=... TAVILY_API_KEY=...
uvicorn app.main:app --reload
# POST http://localhost:8000/research/query  {"query": "..."}  with Authorization: Bearer <token>
```

`app/graph/run_full_stack_demo.py` runs the **real compiled graph**
with mocks only at the three true external SDK boundaries — OpenAI's
`chat.completions.create`, Pinecone's `AsyncIndex.query`, Tavily's
`AsyncTavilyClient.search`. Every line of our own code executes for
real: decomposer, router, both retrievers, grader, synthesizer,
reflection, and the citation validator. This is the strongest
end-to-end proof available without a live API key.

```bash
cd backend
python -m app.graph.run_full_stack_demo
```

Sample output: a query with **deliberate typos**
(`"explian how attension mechnisms work"`) gets corrected by the real
decomposer prompt to `"Explain how attention mechanisms work"`,
retrieves 4 chunks (2 vector + 2 web across 2 sub-questions run
concurrently), and synthesizes a cited answer — all through
unmodified production code.

## Run the frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Set `NEXT_PUBLIC_BACKEND_URL` if the API is not on `http://localhost:8000`.
The UI can also generate a demo JWT if `SUPABASE_JWT_SECRET` matches the backend.

## Test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
cd backend
pytest -v                                                          # 295 tests, ~19s, fully offline
pytest --cov=app --cov-report=term-missing
```

**295 tests. Every real logic file in the entire application is at
100% coverage** — `build.py`, `nodes.py`, `state.py`, `base.py`,
`checkpointing.py`, `loaders.py`, `ingest_store.py`, `ingestion/runner.py`,
`quota_store.py`, `db.py`, `store.py` (99% — one line, a compact
ternary's already-parsed-value branch, directly verified correct in
isolation; see the code comment), and the entire API layer —
`deps.py`, `jwks.py`, `routes.py`, `schemas.py`, `sse.py`, `main.py`,
`runner.py`. `orchestrator.py` is 98% (one trivial unreachable line).
The remaining ~19% overall gap is exclusively: real singleton/client
construction across `client.py` / `embeddings.py` / `pinecone_store.py`
/ `tavily_search.py` that can't be meaningfully tested without live
credentials, the real (non-fallback) `tiktoken`-available branches in
`chunker.py` — genuinely untestable in
this network-restricted sandbox — and the three standalone
demo/smoke-test scripts (not pytest modules).

Test files, mapped to what they prove:

| File | What it proves |
|---|---|
| `test_decomposer_node.py` | Node contract (partial dict return), exception → `state["error"]` |
| `test_router_node.py` | Route selection, `cycle_count` increments correctly across loops |
| `test_grader_node.py` | Threshold filtering, score sorting, `MAX_CHUNKS_TO_SYNTH` cap, **parallel grading via `asyncio.gather`** (timed, not just inferred), graceful empty-list degradation |
| `test_synthesizer_node.py` | **The most important test in the suite** — synthesizer never calls the LLM when `graded` is empty (edge case 3.3, the most dangerous silent failure in the whole system) |
| `test_conditional_edges.py` | `route_after_reflection` hard-caps at `MAX_CYCLES` even if reflection never passes — the guard against reflection thrashing and its 2.7x cost blowup |
| `test_graph_integration.py` | Runs the **real compiled graph**, not isolated functions — proves `operator.add` reducers genuinely accumulate `chunks`/`reflection_history` across cycles, proves topic-less and multi-intent queries never reach retrieval, proves recency queries route web-only |
| `test_guard_and_error_nodes.py` | `quota_check_node`, `error_handler_node` |
| `test_retriever_and_reflection_exceptions.py` | Pinecone failure → `degraded=True` (not a crash), Tavily failure caught, reflection LLM failure **fails open** (treated as pass, not an infinite loop) |
| `test_citation_validation.py` | Edge case 4.1 — hallucinated `[N]` citations stripped, valid ones preserved, citation map only includes indices actually cited |
| `tests/llm/test_client.py` | `call_json`'s retry-then-succeed and give-up-after-3-attempts behavior against the real `tenacity` decorator; JSON repair (markdown fences, trailing commas) against realistic malformed LLM output |
| `test_llm_call_integration.py` | The real `_call_*_llm` functions (not mocked), only the OpenAI client itself faked — proves actual prompt construction, model selection, and XML chunk-wrapping for prompt-injection defense |
| `test_retriever_orchestration.py` | `_call_vector_retriever` / `_call_web_search` fan out one embed+query (or search) **per sub-question concurrently** — timed proof, not inferred; results correctly flattened across sub-questions |
| `tests/retrieval/test_pinecone_store.py` | `PineconeVectorStore.query/upsert/close` against real `pinecone` SDK response types (`QueryResponse`, `ScoredVector`) — namespace passed through unmangled (security-critical), missing metadata handled gracefully |
| `tests/retrieval/test_tavily_search.py` | `search_web()` normalizes Tavily's response shape correctly; **reads the actual source via `inspect.getsource()`** to prove `search_depth="advanced"` is really being passed, not just documented |
| `tests/llm/test_embeddings.py` | `embed_texts()`/`embed_text()` batch ordering and empty-input short-circuit |
| `test_monkeypatch_scoping_regression.py` | Permanent regression test for the `from X import Y` monkeypatch-scoping bug described below |
| `tests/api/test_deps.py` | JWT stub verification, missing/malformed `Authorization` header rejected with 401 |
| `tests/api/test_start_research.py` | `POST /research/query` returns 202 immediately, boundary-length validation (`min_length`/`max_length` exactly), different tokens get isolated runs |
| `tests/api/test_sse_stream.py` | Node events arrive in correct order over a live stream, `citation_map` precedes `run_complete`, stream terminates (doesn't hang) after completion, cross-user stream access blocked |
| `tests/api/test_sse_generator.py` | Unknown `run_id` yields an error event and stops; disconnect detection actually unsubscribes; polling loop continues correctly when no event is ready yet |
| `tests/api/test_get_result.py` | Result reflects true run status pre/post completion, cross-user access blocked, unauthenticated requests rejected |
| `tests/api/test_lifespan.py` | Pinecone warmup failure never crashes app startup |
| `test_store.py` | Replay buffer gives late subscribers full history; multiple independent subscribers each get all events; `owns()` correctly gates cross-user access |
| `test_runner.py` | `stream_mode=["updates","values"]` correctly produces both live events and true final state in one pass; unexpected exceptions never leave a run stuck in `RUNNING` forever |
| `test_checkpointing.py` | **The real resumability proof** — a graph interrupted right after `grader` (before synthesis ever runs) resumes from that exact checkpoint on the same `thread_id`: earlier nodes provably do not re-run, and the final state correctly includes work done before the interruption |
| `tests/api/test_ingest_routes.py` | The three mutually-exclusive input modes (file/url/text) each work; 422 on zero or multiple inputs provided; 413 fast-fail on oversized upload before processing; cross-user job access blocked (404); both `LoaderError` and unexpected exceptions correctly mark a job `failed` rather than leaving it stuck `processing` |
| `test_ingest_store.py` | `JobStore`'s status/result/error/`owns()` logic in isolation, mirroring `test_store.py`'s pattern for `RunStore` |
| `test_quota_store.py` | `QuotaStore`'s usage tracking, per-user isolation, and the has-quota boundary (exactly at, just under, and over the limit) |
| `test_guard_and_error_nodes.py` | Real quota enforcement: a user over their limit is blocked, usage is correctly isolated per user, `formatter_node` AND `error_handler_node` both record real usage (the latter specifically because a rejected query can still have spent real money) |
| `tests/api/test_jwks.py` | `get_jwks()`'s own fetch/cache/TTL-expiry/force-refresh logic via `httpx.MockTransport` — proves the caching actually works, not just that `deps.py` calls whatever it returns |
| `tests/ingestion/test_chunker.py` | Token-based (not character-based) size limits, overlap correctness, orphan-chunk filter, figure-reference stripping, `chunk_text_with_stats`'s accurate dropped-count reporting, and the full `TestTiktokenUnavailableFallback` class proving graceful degradation when `tiktoken`'s network download fails |
| `tests/ingestion/test_loaders.py` | PDF text extraction against **real, `reportlab`-generated PDFs** (not fixtures); 20MB DoS guard rejects before parsing; password-protected PDFs rejected cleanly; a single malformed page doesn't kill extraction of the rest; URL fetching mocked via `httpx.MockTransport` (never real network); HTML boilerplate (`script`/`style`/`nav`/`footer`) stripped |
| `tests/ingestion/test_orchestrator.py` | Idempotent vector IDs on re-ingestion (same source → same IDs, enabling Pinecone overwrite-not-duplicate); `embed_model` stamped into every chunk's metadata; accurate `chunks_dropped` reporting; loader errors propagate without wasting embedding/upsert calls; both `pdf` and `url` source-type success paths |

### Proof this suite actually catches regressions

Six real bugs were caught live while building this, not staged:

1. We deliberately removed the empty-`graded` guard from
   `synthesizer_node` and re-ran the suite — it failed immediately
   with the exact assertion describing edge case 3.3.
2. `run_demo.py`'s first version naively `dict.update()`-ed streamed
   deltas to reconstruct final state, silently under-reporting
   accumulated `chunks` (2 instead of 12) — a bug in the test
   harness itself, not the graph. Fixed by using LangGraph's
   `stream_mode="values"` to get the true reduced state.
3. `test_grader_llm_truncates_chunk_to_500_chars` initially asserted
   on the wrong character (`"x"`, which also appears in
   `"example.com"`) — a false-positive-prone assertion, caught the
   moment the test ran, fixed by using a character (`"z"`)
   guaranteed not to appear elsewhere in the prompt.
4. Building `run_full_stack_demo.py` surfaced a real Python gotcha:
   `app/llm/embeddings.py` does `from app.llm.client import
   get_client`, which creates a **separate local binding** in the
   `embeddings` module's namespace. Patching `app.llm.client.
   get_client` alone silently does nothing for calls made from
   inside `embeddings.py` — the demo failed with a live
   `Missing credentials` error despite every mock supposedly being
   in place. Fixed by patching `app.llm.embeddings.get_client`
   directly, and permanently regression-tested in
   `test_monkeypatch_scoping_regression.py` so this exact class of
   bug can't silently reappear if someone refactors the imports.
5. `chunker.py` called `tiktoken.get_encoding("cl100k_base")` at
   **module import time**, which downloads a vocab file over the
   network on first use — this crashed the entire test suite's
   collection phase the moment `test_chunker.py` was written, since
   this sandbox's network egress allowlist doesn't include the
   download domain. Fixed by lazy-loading with a graceful fallback to
   approximate character-based token counting, matching the same
   degrade-don't-crash pattern used for Pinecone and MLflow elsewhere.
   Permanently regression-tested via `TestTiktokenUnavailableFallback`,
   including a specific test proving a failed download is cached as
   failed and never retried per-chunk (which would otherwise add real,
   compounding latency across every chunk in a large document).
6. `test_load_pdf_survives_a_single_malformed_page`'s first version
   generated its two-page test fixture by calling `canvas.drawString()`
   twice without ever calling `canvas.showPage()` in between — so both
   lines silently landed on a single PDF page instead of two. The test
   failed with a confusing "No extractable text found" error instead
   of testing what it claimed to. Fixed by adding an explicit
   multi-page PDF builder helper (`_make_multipage_pdf_bytes`) that
   calls `showPage()` between pages.
7. Extending the live HTTP smoke test to exercise the new ingest
   routes immediately failed with a real `AttributeError`:
   `FakePineconeIndex` (used across both the full-stack demo and the
   HTTP smoke test) only ever implemented `.query()`, since it
   predates the ingestion pipeline — nobody had exercised the
   `.upsert()` path against it before. The failure surfaced exactly as
   designed: the ingestion job runner's top-level exception handler
   caught it and marked the job `failed` cleanly rather than crashing
   the process, which is itself a good sign, but the fake needed a
   real `.upsert()` method to actually prove the success path works.
8. **This one isn't a code bug — it's a design gap in the project
   itself, caught by direct scrutiny, not by a test failure.**
   `verify_jwt()` was a clearly-labeled stub accepting any non-empty
   string as a valid token. Every test in the suite passed the whole
   time — because every test *used* the stub's permissive behavior
   rather than testing against it. A green test suite proved the
   plumbing worked; it said nothing about whether the plumbing was
   secure. Fixing it required rewriting every test file that used an
   arbitrary `"Bearer some-string"` (which real verification correctly
   rejects as an invalid signature) to mint a properly signed test JWT
   instead — the fix touched far more files than the vulnerability
   itself. This is the most important entry in this list: **a fully
   green test suite is not the same claim as "this is secure,"** and
   the only way to catch that gap was to have someone explicitly ask
   whether the backend was really complete instead of trusting the
   test count.
9. **The same class of gap as #8, this time for cost control.**
   `quota_check_node` always returned `quota_ok=True` — checked at
   exactly the right point in the graph, but the value it checked
   never changed, making it functionally identical to no check at
   all. Fixing it properly required more than just wiring up a real
   store: it required changing `call_json()`/`call_text()`'s return
   contract from a bare payload to `(payload, tokens_used)`, since
   there was previously no path anywhere in the codebase that
   extracted real token counts from an OpenAI response. That single
   contract change cascaded through all four `_call_*_llm` functions,
   every node that calls them, and every test file that mocked any of
   them — a small, correct fix with a large, mechanical blast radius.
10. Enhancing `run_full_stack_demo.py`'s fake OpenAI responses to
    include realistic token counts (so the demo's quota output would
    show real numbers instead of always 0) immediately revealed that
    the script's existing pattern of calling `research_graph.astream()`
    **twice** — once for `stream_mode="updates"` printing, once for
    `stream_mode="values"` — silently double-counted quota usage
    (11,210 recorded instead of the true 5,605), because
    `formatter_node` now has a real side effect. This exact
    double-`astream()` pattern was harmless before this fix (nothing
    it touched had side effects) and became a real bug the moment one
    of its nodes started writing to persistent state — fixed the same
    way `app/graph/runner.py` already solves it: one combined
    `stream_mode=["updates", "values"]` call instead of two separate
    ones.
11. Restoring this project after a sandbox reset surfaced a genuinely
    missing dependency: `python-multipart`, required by FastAPI for
    `Form()`/`File()` handling in the ingest routes, was never listed
    in `requirements.txt` — it happened to already be present in the
    original development environment, so its absence was invisible
    until a truly clean install was attempted. A `pip install -r
    requirements.txt` that "works" in the environment it was written
    in is not proof it's complete; only a fresh environment proves that.
12. **The most subtle bug in this entire project, and a direct
    consequence of the "degrade gracefully, never crash" principle
    applied everywhere else.** `run_http_smoke_test.py` constructed
    `TestClient(app)` without the `with` block Starlette requires to
    actually trigger the ASGI lifespan protocol — meaning the Pinecone
    warmup and (once added) the checkpointer had **never actually run
    during this script's entire existence**. This was completely
    invisible: every route still worked, because both failures are
    designed to degrade non-fatally rather than crash. A silently
    skipped lifespan looked *exactly* like a lifespan that ran and
    degraded — indistinguishable from the outside, until something
    explicitly asserted `is_checkpointing_enabled() is True` and it
    came back `False` with no exception anywhere to point at why.
    Fixed by wrapping the whole script body in `with TestClient(app)
    as client:`. The lesson worth keeping: a design principle that
    makes failures quiet (correct, in general) also makes *test
    harness bugs that prevent a feature from ever running* quiet in
    exactly the same way — the two are indistinguishable without an
    explicit, positive assertion that the thing you expect to have
    happened actually happened.

All twelve are included here deliberately — a test suite (and its
author's understanding of the codebase) is only as trustworthy as
its own bug history is honest.

## What's proven to work right now

1. The fan-out at the Router (`pick_sources` returning a list) runs
   `vector_retriever` and `web_search` in the same LangGraph
   superstep — true parallel execution, not sequential.
2. The implicit join at `grader` correctly waits for both branches
   before running once.
3. `chunks` accumulates correctly across reflection cycles via
   `operator.add` (verified: 12 chunks after 3 cycles × 2 sources ×
   2 sub-questions — not silently overwritten).
4. `reflection_history` accumulates one entry per cycle, giving the
   Router real history to use for rewriting sub-questions (the gap
   string in each entry is exactly what would feed back into the
   Router to prevent reflection thrashing).
5. `route_after_reflection` hard-caps at `MAX_CYCLES = 3` regardless
   of what reflection says — the single most important guard against
   runaway cost.
6. A topic-less query (`topic_identified: False`) is rejected by
   `check_decomposition` before touching `vector_retriever`,
   `web_search`, `grader`, or `synthesizer` — verified by node
   visitation list, not just by reading the code.
7. The complete HTTP + SSE flow works end-to-end against the real
   FastAPI app: unauthenticated requests rejected (401), a research
   run started via `POST /research/query` returns instantly with a
   `run_id`, cross-user access to another user's run is blocked
   (404), and every graph node's completion streams as an ordered SSE
   event ending in `citation_map` then `run_complete` — verified via
   `run_http_smoke_test.py` driving the real ASGI app, not mocked
   route handlers.
8. Pinecone/Tavily failures degrade gracefully at the HTTP layer too
   — a failed run publishes an `error` SSE event and sets status to
   `failed` rather than leaving the client's stream connection
   hanging forever with no signal.
9. The full ingestion pipeline works end-to-end against real,
   `reportlab`-generated PDF structure — text extraction, chunking,
   idempotent vector-ID generation, and upsert into a fake
   `VectorStore` all verified together, not just at the unit level.
   Re-ingesting the identical source twice produces identical vector
   IDs (the idempotency guarantee), and a document engineered to
   produce a genuine dropped orphan chunk correctly reports
   `chunks_dropped >= 1` instead of the silently-wrong `0` the code
   originally always returned.

## Next integration steps

1. Real Pinecone/Tavily/OpenAI wiring is done, the ingestion pipeline
   is fully built and tested, **auth now performs real JWT
   verification**, **quota now tracks real per-user token usage**, and
   **runs are now checkpointed and provably resumable within a running
   process** (see all three fix sections above). Every previously-flagged
   stub in the core request path is closed.
2. Replace `RunStore`, `JobStore`, and `QuotaStore`'s in-memory dicts
   with Supabase-backed implementations behind the same interfaces
   (`create_run`/`publish_event`/`subscribe`/`get_result`/`owns`,
   `create_job`/`set_result`/`set_error`/`owns`,
   `has_quota`/`add_usage`/`get_usage`) — every route, runner, and
   node already depends only on those interfaces, not on any store's
   internals, so this is a swap, not a rewrite. This is also what
   closes the checkpointer's scope caveat above: once `RunStore`
   itself survives a process restart, a genuinely crashed-and-restarted
   FastAPI process can reconnect a client's SSE stream to a run that
   was mid-flight and call `resume_graph_and_publish()` for it.
3. MLflow observability — designed in detail earlier in this project
   (per-node latency/token/cost metrics, prompt-version comparison)
   but never implemented. `mlflow_run_id` exists in `ResearchState`
   as a placeholder UUID; no `mlflow.log_params`/`log_metrics`/
   `log_artifact` calls exist anywhere yet.
4. Docker Compose for this actual codebase, and a deployment target.
5. A minimal Next.js frontend in `frontend/` — enough to demo the SSE
   streaming and ingestion flow, not production-styled.
