# parallax_backend

Go 1.26 service for Parallax: the Director agent loop, project/timeline storage, media
indexing, and export. Project-wide docs (audit, roadmap, ADRs) are in `../docs/`.

## Commands

```bash
go build ./...                 # build
go vet ./internal/...          # lint (one known warning: projects/store.go mutex copy)
go test ./...                  # all tests
go test ./internal/tools -run Audio -v   # one package / test
cp .env.example .env && go run ./cmd/server   # run (needs Postgres, Qdrant, Supabase settings)
docker compose up -d           # Postgres + Qdrant + backend (compose.nvidia.yaml for GPU)
```

Known on Windows: 8 tests fail because they assume Unix paths, symlink rights, Python or
network access (same on upstream). `internal/auth` and `tests/qdrant` are occasionally flaky
under full parallel runs. Compare against `upstream/main` before blaming a change.

## Architecture

- `cmd/server`: wiring and startup. `internal/httpapi`: routes, SSE chat
  (`POST /v1/agent/chat`), tus uploads, export. Each chat request builds a tool registry in
  `handleChat`.
- `internal/agent`: framework-free loop (think → tools → observe), system prompt in `prompt.go`.
- `internal/tools`: one file per tool group, registered with `Register<Name>(reg, Env)`.
  Tools that change media call `OnMutation` + `OnApplied(rel)` so files are published and
  indexed.
- `internal/projects`: timeline schema v3 (integer frames), validated operations, revisions;
  each Director request is one `TimelineTransaction` committed as one revision.
- `internal/ffmpeg`: every FFmpeg call goes through `Validate` (no shell, workspace-only paths,
  no network). Never bypass it.
- `internal/transcript`: background indexer (whisper → translate → embed → Qdrant).

## Conventions

- Remotes: `origin` = Yash-Satankar fork, `upstream` = Rohit-RA-2020 (push disabled). Bring in
  upstream with `git fetch upstream && git merge upstream/main`.
- Never log API keys, tokens or signed URLs. Keys must never be serialised to clients
  (`api_key_set` only).
- Commit messages: `feat(scope): …`, `fix(scope): …`. One branch per phase; never force-push
  `main`.
