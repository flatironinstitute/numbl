# Execution Server

An optional standalone HTTP service that runs numbl scripts on behalf of remote clients (typically the web IDE when native performance is wanted).

## Role

- Accepts a request describing the files to run, the main script, and an optimization level.
- Runs the script via `numbl run --stream` as a child process in a temporary working directory, with stdin closed (so `input()` returns an empty value).
- Relays the CLI's stream messages (see [cli.md](cli.md#stream-protocol)) as server-sent events, with file paths made relative to the project, then appends a `close` message. The IDE consumes them through `RemoteExecutionBackend` ([web-app.md](web-app.md#execution-backends)).
- Enforces per-request limits: concurrency, execution timeout, memory cap.
- Gates access behind a shared-secret passkey header.

It is not part of the library surface — it is a separate deployable.

## Why it exists

The browser worker has no LAPACK/FFTW addon and no C-JIT (`--opt 2`). For workloads where that matters, the web IDE can optionally offload execution to a server that does have those things installed. The same `.m` files run in both places; the server is just a faster backend.

## Deployment notes

The server is expected to be run under a process supervisor (a PM2 ecosystem file is included). It spawns the CLI per request in an isolated temp directory; it does not share state across requests.
