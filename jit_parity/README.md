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
  fails to error where the interpreter does). Includes value bugs (e.g. reduce
  along a trailing singleton axis), type-class loss (a logical array typed as
  double), inconsistent dim/rep coercion (round vs truncate vs validate),
  complex-transcendental edge cases, and reads of never-assigned locals/outputs.
- **B** — whole-scope/loop JIT runs side effects, then bails, and the interpreter
  re-runs → duplicated output.
- **C** — interpreter used the native `-ffast-math` LAPACK addon while the JITs
  use their own kernels → different float results / NaN handling. Resolved by
  defaulting the addon build to **no-fast-math** (deterministic reductions and
  transcendentals; opt in with `--fast-math`), plus a NaN-correct any/all.
- **D** — stdout pollution from a debug log (now on stderr).
- **E** — C-JIT (`--opt 2`) **fails to compile** a program the interpreter and
  JS-JIT run fine (e.g. a `setjmp` grow-bail guard armed without including
  `<setjmp.h>`, or a statically-scalar range slice typed scalar but emitted as a
  tensor). Surfaces as `<ERROR>` only at opt2.
- **F** — C-JIT `printf`/`disp` **formatting** diverges from the JS path (libc
  `snprintf` rounds half-to-even vs JS half-away; non-finite spelling
  `Infinity`/`inf` vs `Inf`; large integer-valued doubles; `%s`/`%x`/`%o` edges).
- **G** — struct/cell `disp` writes through raw `printf`/`fputs(stdout)` instead
  of the host-write callback the rest of output uses → reordering at opt2 (and
  lost output under captured/streamed runs).

## Gate exclusions

Some scripts are **excluded from the pass/fail gate** (they still run and show
as `EXCL`, with the reason printed, but don't affect the exit code). See
`GATE_EXCLUDED` in `run.mjs`. Two kinds:

- **Inherent FP / library-ULP differences** — can't be made bitwise-identical
  without dropping BLAS/libm:
  - `C03 matmul` — BLAS `dgemm` (interpreter) vs a naive triple loop (JIT)
    accumulate in different orders → last-bit differences.
  - `A35 complex-pow` — `(-1)^(0.5+1i)` keeps a `~2.6e-18` real residue
    (`exp(-π)·cos(π/2)`); MATLAB keeps it too, but it's the difference of
    transcendentals, so opt2 (libm) and opt0/opt1 (V8) disagree at ULP scale
    (same class as `C02`). The integer-power line *is* fixable; the script
    as a whole can't be byte-stable.

- **Needs a V8-equivalent dtoa in C (deferred)** — the C format engine's
  number→string differs from V8's, and a naive fix would pass these
  exact-representable cases while introducing *new* divergences for non-exact
  values:
  - `F01` / `F02` — `%f/%e/%g` and `disp()` scalar half-way rounding: libc
    `snprintf` rounds half-to-even; V8's `toFixed`/`toExponential` correct-round
    toward +Inf on the exact decimal.
  - `F04` — `%s` of a non-integer needs V8's shortest round-trip form
    (`String(0.1)` → `"0.1"`); libc `%.17g`/`%g` aren't shortest.

Everything else in the corpus is expected to PASS.
