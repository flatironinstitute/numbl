/**
 * MATLAB-style warnings from helper code that has no Runtime handle.
 *
 * A message goes to the active runtime's output ("Warning: <msg>"); with no
 * runtime active it falls back to console.warn.
 */

import { getCurrentRuntime } from "../runtime/refcount.js";

export function emitWarning(msg: string): void {
  const rt = getCurrentRuntime() as {
    output?: (text: string) => void;
  } | null;
  if (rt?.output) rt.output("Warning: " + msg + "\n");
  else console.warn("Warning: " + msg);
}
