// JS sibling of `fprintf.h`. Routes formatted output through
// `jit_sprintf_format` then writes via `$write` (the JS analog of
// C's stdout). Numbl-compatible spec set & semantics.

import { jit_sprintf_format } from "./format_engine.js";

export function jit_fprintf(fmt, ...args) {
  $write(jit_sprintf_format(fmt, args));
}
