/**
 * MATLAB-style warnings from helper code that has no Runtime handle.
 *
 * The Runtime installs a sink that routes messages through its output
 * channel ("Warning: <msg>"); before a Runtime exists, messages fall back
 * to console.warn.
 */

let sink: (msg: string) => void = msg => console.warn("Warning: " + msg);

export function setWarningSink(fn: (msg: string) => void): void {
  sink = fn;
}

export function emitWarning(msg: string): void {
  sink(msg);
}
