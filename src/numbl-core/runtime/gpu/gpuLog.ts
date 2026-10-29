/**
 * NUMBL_GPU_LOG: count the transfers of gpuArray data between host and
 * device by site and source line, and print a summary to stderr at exit.
 * A site is what moved the data: `gather` and `gpuArray` are explicit;
 * a builtin's name, `index`, `assign`, `condition`, `scalar`, `for`, ...
 * mean an operation that ran on the host (a fallback), whose gathers and
 * uploads are each a device round trip; `host` rows are the time such a
 * builtin spent computing on the host. A gather's time includes waiting
 * for the queued work that computes its data.
 */

import { getCurrentRuntime } from "../refcount.js";

export const gpuLogEnabled =
  typeof process !== "undefined" && !!process.env?.NUMBL_GPU_LOG;

interface Entry {
  dir: Dir;
  site: string;
  where: string;
  count: number;
  elems: number;
  ms: number;
}

type Dir = "gather" | "upload" | "host";

const entries = new Map<string, Entry>();

/** Record a transfer of `elems` elements (or host work) that took `ms`. */
export function logTransfer(
  dir: Dir,
  site: string,
  elems: number,
  ms: number
): void {
  if (entries.size === 0) process.once("exit", printSummary);
  const rt = getCurrentRuntime() as { $file?: string; $line?: number } | null;
  const file = rt?.$file ? rt.$file.split("/").pop() : "";
  const where = rt?.$line ? `${file}:${rt.$line}` : "";
  const key = `${dir}\u0000${site}\u0000${where}`;
  let e = entries.get(key);
  if (!e)
    entries.set(key, (e = { dir, site, where, count: 0, elems: 0, ms: 0 }));
  e.count++;
  e.elems += elems;
  e.ms += ms;
}

function printSummary(): void {
  const rows = [...entries.values()].sort((a, b) => b.ms - a.ms);
  const total = rows.reduce((s, e) => s + e.ms, 0);
  const lines = [
    `numbl gpu transfers and host fallbacks (NUMBL_GPU_LOG): ${total.toFixed(1)} ms in all`,
    "      ms   count     elements  dir     site            where",
  ];
  for (const e of rows)
    lines.push(
      `${e.ms.toFixed(1).padStart(8)} ${String(e.count).padStart(7)} ${String(e.elems).padStart(12)}  ${e.dir.padEnd(7)} ${e.site.padEnd(15)} ${e.where}`
    );
  process.stderr.write(lines.join("\n") + "\n");
}
