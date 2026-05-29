// JS sibling of `tictoc.h`. Same "one global timer" semantics. Uses
// `performance.now()` to match numbl's wall-clock source; the output
// format string mirrors numbl's `"Elapsed time is %.6f seconds.\n"`
// so the cross-runner's masking regex still treats both runtimes as
// equivalent.
//
// Module-scope state for the shared `tic` slot. The codegen path
// inlines this declaration alongside the helpers; the interpreter
// imports them as ES exports and the same module-scope `let` is the
// canonical slot.

let mtoc2_tic_seconds = 0;

function mtoc2_now_seconds() {
  // `performance.now()` returns milliseconds since some monotonic
  // epoch; divide for seconds. (Node and browsers both supply it on
  // `globalThis.performance` as of recent versions.)
  return performance.now() / 1000;
}

export function mtoc2_tic() {
  mtoc2_tic_seconds = mtoc2_now_seconds();
  return mtoc2_tic_seconds;
}

export function mtoc2_toc() {
  return mtoc2_now_seconds() - mtoc2_tic_seconds;
}

export function mtoc2_toc_handle(start_seconds) {
  return mtoc2_now_seconds() - start_seconds;
}

export function mtoc2_toc_print() {
  const elapsed = mtoc2_toc();
  $write(`Elapsed time is ${elapsed.toFixed(6)} seconds.\n`);
}

export function mtoc2_toc_handle_print(start_seconds) {
  const elapsed = mtoc2_now_seconds() - start_seconds;
  $write(`Elapsed time is ${elapsed.toFixed(6)} seconds.\n`);
}
