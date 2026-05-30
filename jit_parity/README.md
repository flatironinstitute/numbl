# jit_parity

A corpus of small `.m` scripts that should produce **identical output** under all
three numbl execution modes, plus a runner that checks it:

- `--opt 0` — interpreter (reference / ground truth)
- `--opt 1` — JS-JIT
- `--opt 2` — C-JIT

All three are the same codebase, so an in-bounds run must be byte-for-byte
identical. Every script here is a *known divergence* between modes — i.e. a bug.
As we fix the JIT, scripts flip from `FAIL` to `PASS`; the goal is all `PASS`.

This is intentionally **separate from `numbl_test_scripts/`** (those are
MATLAB-conformance integration tests run in CI). This corpus targets cross-mode
parity specifically.

## Running

```bash
node jit_parity/run.mjs            # all scripts
node jit_parity/run.mjs A04 A17    # only names matching a filter
node jit_parity/run.mjs -v         # full per-mode output on failures
node jit_parity/run.mjs --no-opt2  # skip the slower compiling C-JIT mode
```

Exit code is 0 iff every selected script passes.

## How comparison works

We expect **exact** matches. The runner reduces each mode's run to a comparison
key and requires all three keys equal:

- stdout, compared verbatim, except: lines containing `using bridge:` (a one-time
  native-addon diagnostic) and lines starting with `warning:` (JIT bail notices)
  are dropped, and trailing whitespace is trimmed.
- a non-zero exit collapses to the token `<ERROR>` — the interpreter and C-JIT
  word their error messages differently, so we test the *outcome* ("it correctly
  refused"), not the message text.
- a timeout collapses to `<TIMEOUT>`.

## The corpus

Each filename is prefixed by category. Header comments in each `.m` file record
the diagnosis and the divergence observed at discovery time.

- **A** — genuine JIT correctness bugs (a JIT mode produces a wrong value, or
  fails to error where the interpreter does).
- **B** — whole-scope/loop JIT runs side effects, then bails, and the interpreter
  re-runs → duplicated output.
- **C** — interpreter uses the native `-ffast-math` LAPACK addon while the JITs
  use their own kernels → different float results / NaN handling. Parity here
  means making all three modes agree (deterministic across modes).
- **D** — stdout pollution from a debug log (filtered by the runner; root cause
  still worth removing).
