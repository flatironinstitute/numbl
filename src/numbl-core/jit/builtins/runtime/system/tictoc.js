// JS sibling of `tictoc.h`. Same "one global timer" semantics. Uses
// `performance.now()` to match numbl's wall-clock source; the output
// format string mirrors numbl's `"Elapsed time is %.6f seconds.\n"`.
//
// Module-scope state for the shared `tic` slot. The codegen path
// inlines this declaration alongside the helpers; the interpreter
// imports them as ES exports and the same module-scope `let` is the
// canonical slot.

let jit_tic_seconds = 0;

function jit_now_seconds() {
  // `performance.now()` returns milliseconds since some monotonic
  // epoch; divide for seconds. (Node and browsers both supply it on
  // `globalThis.performance` as of recent versions.)
  return performance.now() / 1000;
}

export function jit_tic() {
  jit_tic_seconds = jit_now_seconds();
  return jit_tic_seconds;
}

export function jit_toc() {
  return jit_now_seconds() - jit_tic_seconds;
}

export function jit_toc_handle(start_seconds) {
  return jit_now_seconds() - start_seconds;
}

export function jit_toc_print() {
  const elapsed = jit_toc();
  $write(`Elapsed time is ${elapsed.toFixed(6)} seconds.\n`);
}

export function jit_toc_handle_print(start_seconds) {
  const elapsed = jit_now_seconds() - start_seconds;
  $write(`Elapsed time is ${elapsed.toFixed(6)} seconds.\n`);
}
