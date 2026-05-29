// Ambient declarations for host hooks that runtime .js snippets
// reference as free variables. The host (CLI, browser, codegen-
// emitted `run($h)`) assigns these on `globalThis` before any
// snippet function runs.
//
// Adding a hook: declare it here, assign it in every host's
// bootstrap, and let snippets reference it by bare name.

declare global {
  /** Append text to stdout. No implicit newline. */
  var $write: (s: string) => void;
}

export {};
