# Starter snapshot: timelens backend

This is a snapshot of a real production Python backend (FastAPI +
SQLModel + Postgres + Redis + a background worker pool), lifted from a
specific commit of the timelens codebase.

`/app` holds the full backend tree — `db/`, `lib/`, `worker/`,
`route_handler/`, `tests/`, etc. — flattened from the original
`backend/` directory so imports resolve from `/app`.

Pytest + dbmate + codegen scripts are all wired up. The test fixtures
start Postgres and Redis on demand. `run_servers.sh` is a no-op.

See `instructions.md` for the task itself.
