/**
 * Host bag passed to `Builtin.call` for the interpreter backend.
 *
 * Mirrors nexl's `RuntimeContext`. Side-effecty builtins (`disp`,
 * `fprintf`, `error`, plot dispatch, …) consult `helpers.write` to
 * emit text. Pure builtins ignore the bag.
 *
 * `write` is the JS analogue of C's `printf`. The CLI provides one
 * backed by `process.stdout.write`; the browser provides one that
 * pushes lines to the console panel; future test runners provide
 * one that captures into a buffer for byte-for-byte comparison.
 *
 * As the interpreter grows, additional hooks (`time`, `plot`,
 * `error`, …) will join this bag. Adding a hook here is the
 * single source of truth — every host supplies it.
 */
export interface RuntimeHelpers {
  /** Append text to stdout. No implicit newline. */
  write(s: string): void;
}

export interface RuntimeContext {
  helpers: RuntimeHelpers;
}
