/**
 * Profile output for `--profile <file>` and the `show-profile` command that
 * pretty-prints it.
 */

import { existsSync, readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import type { ProfileData } from "./numbl-core/executeCode.js";

/** Write the profile JSON consumed by `show-profile`. */
export function writeProfile(
  file: string,
  pd: ProfileData,
  totalTimeMs: number
): void {
  const builtins = Object.entries(pd.builtins)
    .map(([name, b]) => ({
      name,
      fallback: b.fallback,
      interp: b.interp,
      jit: b.jit,
    }))
    .sort((a, b) => {
      const aTotal =
        a.fallback.totalTimeMs + a.interp.totalTimeMs + a.jit.totalTimeMs;
      const bTotal =
        b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs;
      return bTotal - aTotal;
    });
  const dispatches = Object.entries(pd.dispatches)
    .map(([name, b]) => ({
      name,
      totalTimeMs: b.totalTimeMs,
      callCount: b.callCount,
    }))
    .sort((a, b) => b.totalTimeMs - a.totalTimeMs);
  const profile = {
    totalTimeMs,
    executionTimeMs: pd.executionTimeMs,
    jitCompileTimeMs: pd.jitCompileTimeMs,
    builtins,
    dispatches,
    hotLoops: pd.hotLoops ?? [],
  };
  writeFileSync(file, JSON.stringify(profile, null, 2) + "\n");
  console.error(`Profile written to ${file}`);
}

export function cmdShowProfile(args: string[]): void {
  if (args.length === 0) {
    console.error("Error: show-profile requires a profile JSON file argument");
    process.exit(1);
  }
  const filepath = resolve(process.cwd(), args[0]);
  if (!existsSync(filepath)) {
    console.error(`Error: profile file not found: ${filepath}`);
    process.exit(1);
  }
  const data = JSON.parse(readFileSync(filepath, "utf-8"));

  const fmt = (n: number, decimals = 1) => n.toFixed(decimals);
  const pad = (s: string, w: number) => s.padStart(w);

  const execTime = data.executionTimeMs || 1;
  const jitTime = data.jitCompileTimeMs ?? 0;

  // Compute disjoint breakdown of execution time
  const dispatches: { name: string; totalTimeMs: number; callCount: number }[] =
    data.dispatches ?? [];
  const builtins: {
    name: string;
    fallback: { totalTimeMs: number; callCount: number };
    interp: { totalTimeMs: number; callCount: number };
    jit: { totalTimeMs: number; callCount: number };
  }[] = data.builtins ?? [];
  const sumDispatches = dispatches.reduce((s, d) => s + d.totalTimeMs, 0);
  const sumBuiltins = builtins.reduce(
    (s, b) =>
      s + b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs,
    0
  );
  const otherTime =
    data.executionTimeMs - jitTime - sumDispatches - sumBuiltins;

  console.log(`Total time:      ${fmt(data.totalTimeMs)} ms`);
  console.log(`Execution time:  ${fmt(data.executionTimeMs)} ms`);
  console.log(
    `  JIT compile:   ${fmt(jitTime)} ms (${fmt((jitTime / execTime) * 100)}%)`
  );
  console.log(
    `  Dispatches:    ${fmt(sumDispatches)} ms (${fmt((sumDispatches / execTime) * 100)}%)`
  );
  console.log(
    `  Builtins:      ${fmt(sumBuiltins)} ms (${fmt((sumBuiltins / execTime) * 100)}%)`
  );
  console.log(
    `  Other:         ${fmt(otherTime)} ms (${fmt((otherTime / execTime) * 100)}%)`
  );
  console.log("\nAll timings are self-time (exclusive) and mutually disjoint.");

  // Dispatched user functions
  if (dispatches.length > 0) {
    const nameW = Math.max(20, ...dispatches.map(b => b.name.length)) + 2;
    const header =
      "Function".padEnd(nameW) +
      pad("Calls", 10) +
      pad("Self (ms)", 13) +
      pad("Avg (ms)", 11) +
      pad("% Exec", 9);
    console.log("\nDispatched functions (user-defined / workspace):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const b of dispatches) {
      const avg = b.callCount > 0 ? b.totalTimeMs / b.callCount : 0;
      const pct = (b.totalTimeMs / execTime) * 100;
      const line =
        b.name.padEnd(nameW) +
        pad(String(b.callCount), 10) +
        pad(fmt(b.totalTimeMs, 2), 13) +
        pad(fmt(avg, 3), 11) +
        pad(fmt(pct, 1) + "%", 9);
      console.log(line);
    }
  }

  // Builtin functions
  if (builtins.length > 0) {
    const nameW = Math.max(20, ...builtins.map(b => b.name.length)) + 2;
    const header =
      "Builtin".padEnd(nameW) +
      pad("Interp", 10) +
      pad("Fallback", 10) +
      pad("JIT", 10) +
      pad("Self (ms)", 13) +
      pad("% Exec", 9);
    console.log("\nBuiltin functions (call counts by source):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const b of builtins) {
      const totalTime =
        b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs;
      const pct = (totalTime / execTime) * 100;
      const line =
        b.name.padEnd(nameW) +
        pad(String(b.interp.callCount), 10) +
        pad(String(b.fallback.callCount), 10) +
        pad(String(b.jit.callCount), 10) +
        pad(fmt(totalTime, 2), 13) +
        pad(fmt(pct, 1) + "%", 9);
      console.log(line);
    }
  } else {
    console.log("\nNo builtin functions were called.");
  }

  // Hot interpreted loops (>1000 iterations)
  const hotLoops: {
    file: string;
    line: number;
    kind: string;
    iterations: number;
    callCount: number;
    totalTimeMs: number;
  }[] = data.hotLoops ?? [];
  if (hotLoops.length > 0) {
    hotLoops.sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    const locW =
      Math.max(20, ...hotLoops.map(l => `${l.file}:${l.line}`.length)) + 2;
    const header =
      "Location".padEnd(locW) +
      pad("Kind", 8) +
      pad("Calls", 8) +
      pad("Max Iters", 12) +
      pad("Total (ms)", 13);
    console.log("\nInterpreted loops (>1000 iterations):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const l of hotLoops) {
      const loc = `${l.file}:${l.line}`;
      const line =
        loc.padEnd(locW) +
        pad(l.kind, 8) +
        pad(String(l.callCount ?? 1), 8) +
        pad(String(l.iterations), 12) +
        pad(fmt(l.totalTimeMs ?? 0, 2), 13);
      console.log(line);
    }
  }
}
