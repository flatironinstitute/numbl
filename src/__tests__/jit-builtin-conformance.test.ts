/// <reference types="node" />

/**
 * Per-builtin conformance between the interpreter (`--opt 0`) and the JIT
 * backends (`--opt 1`, and `--opt 2` when `cc` and koffi are available).
 * Every registered JIT builtin runs its case table (`jit-conformance/
 * cases.ts`) through compiled kernels, and each case that compiled must
 * produce the interpreter's result exactly: class, shape, complexness,
 * every element bit for bit (NaN payloads aside), its printed output and
 * its error. See `jit-conformance/harness.ts` for how a case runs.
 *
 * `NUMBL_CONFORMANCE_VERBOSE=1` prints per-builtin counts and the decline
 * reasons; `NUMBL_CONFORMANCE_DUMP=<file>` saves the last generated
 * script, to rerun with the CLI.
 */

import { describe, it, expect, afterAll } from "vitest";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import {
  allBuiltinNames,
  getBuiltin,
} from "../numbl-core/jit/builtins/index.js";
import { registerNodeCompileC } from "../numbl-core/executors/jit/cjit.node.js";
import type { NativeBridge } from "../numbl-core/workspace/types.js";
import {
  caseId,
  explains,
  runBatched,
  runSingle,
  type Case,
  type Outcome,
} from "./jit-conformance/harness.js";
import {
  ALLOWLIST,
  KNOWN_DIVERGENCES,
  NO_JS_JIT,
  TABLES,
  familyCases,
} from "./jit-conformance/cases.js";

registerNodeCompileC();

/** The koffi bridge when C-JIT can run here, else undefined. */
function cBridge(): NativeBridge | undefined {
  try {
    execFileSync("cc", ["--version"], { stdio: "ignore" });
    const koffi = createRequire(import.meta.url)("koffi");
    return { load: (path: string) => koffi.load(path), koffi };
  } catch {
    return undefined;
  }
}
const bridge = cBridge();
const VERBOSE = !!process.env.NUMBL_CONFORMANCE_VERBOSE;

const registered = [...allBuiltinNames()].sort();

function tableOf(name: string): Case[] {
  const b = getBuiltin(name);
  if (!b) throw new Error(`'${name}' is not a registered JIT builtin`);
  return [...familyCases(b), ...(TABLES[name] ?? [])];
}

function same(a: Outcome, b: Outcome): boolean {
  if (a.kind === "value" && b.kind === "value")
    return (
      a.output === b.output &&
      a.values.length === b.values.length &&
      a.values.every((v, k) => v === b.values[k])
    );
  if (a.kind === "error" && b.kind === "error")
    return a.error === b.error && a.output === b.output;
  return false;
}

function show(o: Outcome): string {
  switch (o.kind) {
    case "value":
      return (
        o.values.join(" ; ") +
        (o.output ? ` [output ${JSON.stringify(o.output)}]` : "")
      );
    case "error":
      return (
        `error ${o.error}` +
        (o.output ? ` [output ${JSON.stringify(o.output)}]` : "")
      );
    case "declined":
      return `declined: ${o.reason}`;
  }
}

interface LevelStats {
  matched: number;
  declined: number;
  known: number;
}
interface Row {
  name: string;
  cases: number;
  levels: Partial<Record<"1" | "2", LevelStats>>;
  declines: string[];
}
const rows: Row[] = [];

/** Compare one JIT level against the reference; returns failure lines.
 *  `used` collects the known divergences (and their case ids) that
 *  explained a mismatch. */
function check(
  name: string,
  cases: Case[],
  ref: Outcome[],
  got: Outcome[],
  opt: "1" | "2",
  row: Row,
  used: Set<string>
): string[] {
  const stats: LevelStats = { matched: 0, declined: 0, known: 0 };
  const failures: string[] = [];
  const known = KNOWN_DIVERGENCES.filter(
    d => d.opts.includes(opt) && d.names.includes(name)
  );
  got.forEach((o, i) => {
    const id = caseId(name, cases[i]);
    if (o.kind === "declined") {
      stats.declined++;
      row.declines.push(`opt${opt} ${id}: ${o.reason}`);
      return;
    }
    if (same(ref[i], o)) {
      stats.matched++;
      return;
    }
    const d = known.find(k => explains(k, id, ref[i], o));
    if (d) {
      stats.known++;
      used.add(d.kind === "case" ? `${d.reason}|${id}` : d.reason);
      return;
    }
    failures.push(
      `--opt ${opt} ${id}\n    opt0: ${show(ref[i])}\n    opt${opt}: ${show(o)}`
    );
  });
  row.levels[opt] = stats;
  return failures;
}

/** Known divergences listed for `name` that no longer occur. */
function staleKnown(name: string, ran: string[], used: Set<string>): string[] {
  const stale: string[] = [];
  for (const d of KNOWN_DIVERGENCES) {
    if (
      d.optional ||
      !d.names.includes(name) ||
      !d.opts.every(o => ran.includes(o))
    )
      continue;
    const keys =
      d.kind === "case" ? d.ids.map(id => `${d.reason}|${id}`) : [d.reason];
    for (const k of keys)
      if (!used.has(k)) stale.push(`known divergence no longer seen: ${k}`);
  }
  return stale;
}

describe("JIT builtin conformance", () => {
  it("every registered JIT builtin has a case table or an allowlist entry", () => {
    const missing = registered.filter(
      n => !(n in ALLOWLIST) && tableOf(n).length === 0
    );
    const both = registered.filter(
      n => n in ALLOWLIST && tableOf(n).length > 0
    );
    const stale = [
      ...Object.keys(TABLES),
      ...Object.keys(ALLOWLIST),
      ...Object.keys(NO_JS_JIT),
    ].filter(n => getBuiltin(n) === undefined);
    expect({ missing, both, stale }).toEqual({
      missing: [],
      both: [],
      stale: [],
    });
  });

  for (const name of registered) {
    if (name in ALLOWLIST) continue;
    it(name, { timeout: 300_000 }, () => {
      const cases = tableOf(name);
      const row: Row = { name, cases: cases.length, levels: {}, declines: [] };
      rows.push(row);
      const used = new Set<string>();
      const ref = runSingle(name, cases, "0");
      const got1 = runSingle(name, cases, "1");
      const failures = check(name, cases, ref, got1, "1", row, used);
      const ran = ["1"];
      if (bridge) {
        // Batch the cases that ran without error at --opt 0.
        const got2 = runBatched(
          name,
          cases,
          cases.map((c, i) => ref[i].kind === "value" && !c.pre),
          "2",
          bridge
        );
        failures.push(...check(name, cases, ref, got2, "2", row, used));
        ran.push("2");
      }
      failures.push(...staleKnown(name, ran, used));
      expect(failures).toEqual([]);
      // A table where nothing compiles checks nothing.
      const compiled = (opt: "1" | "2") =>
        (row.levels[opt]?.matched ?? 0) + (row.levels[opt]?.known ?? 0);
      expect(compiled("1") > 0).toBe(!(name in NO_JS_JIT));
    });
  }

  afterAll(() => {
    const total = (opt: "1" | "2", k: keyof LevelStats) =>
      rows.reduce((s, r) => s + (r.levels[opt]?.[k] ?? 0), 0);
    const cases = rows.reduce((s, r) => s + r.cases, 0);
    const lines = [
      `JIT builtin conformance: ${rows.length} builtins with case tables, ` +
        `${Object.keys(ALLOWLIST).length} allowlisted, ` +
        `${registered.length} registered; ${cases} cases`,
    ];
    for (const opt of ["1", "2"] as const) {
      if (!rows.some(r => r.levels[opt])) {
        lines.push(`  --opt ${opt}: skipped (needs cc and koffi)`);
        continue;
      }
      const none = rows.filter(
        r => r.levels[opt] && r.levels[opt]!.matched === 0
      );
      lines.push(
        `  --opt ${opt}: ${total(opt, "matched")} compiled and matched, ` +
          `${total(opt, "declined")} declined, ${total(opt, "known")} known divergences; ` +
          `nothing compiled for: ${none.map(r => r.name).join(" ") || "-"}`
      );
    }
    if (VERBOSE) {
      for (const r of rows) {
        const lv = (opt: "1" | "2") =>
          r.levels[opt] ? `${r.levels[opt]!.matched}/${r.cases}` : "-";
        lines.push(`  ${r.name}: opt1 ${lv("1")}, opt2 ${lv("2")}`);
        for (const d of r.declines) lines.push(`      ${d}`);
      }
    }
    console.log(lines.join("\n"));
  });
});
