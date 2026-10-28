/**
 * Result format, markdown summary and comparison for benchmarks/bench.ts.
 */

export interface Entry {
  case: string;
  mode: string;
  size: number;
  status: "ok" | "skipped" | "error";
  /** Median per-call time over the timed batches. */
  median_ms?: number;
  min_ms?: number;
  /** Wall time of the first call at this size, measured outside the
   *  kernel: includes JIT compilation when it happens. */
  first_ms?: number;
  /** Kernel calls per timed batch. */
  inner?: number;
  /** Number of timed batches. */
  reps?: number;
  /** sum(abs(y(:))) of the last result, for cross-mode sanity checks. */
  checksum?: number;
  error?: string;
}

export interface BenchResult {
  schema: 1;
  created: string;
  machine: {
    cpu: string;
    cores: number;
    memGB: number;
    os: string;
    node: string;
    numblVersion: string;
    commit: string;
    dirty: boolean;
    nativeAddon: boolean;
    cc?: string;
    koffi: boolean;
    env: Record<string, string>;
    /** 1-minute load average at the start and end of the run: a busy
     *  machine makes the timings noisy. */
    loadavg?: [number, number];
  };
  config: Record<string, unknown>;
  totalWallS: number;
  modes: { name: string; args: string; startup_ms: number }[];
  cases: { name: string; description: string; sizeLabel: string }[];
  entries: Entry[];
  /** JIT declines, bail warnings and other non-benchmark output, per case
   *  and mode (deduplicated). */
  diagnostics: { case: string; mode: string; lines: string[] }[];
}

export function fmtSize(n: number): string {
  const l10 = Math.log10(n);
  if (n >= 1000 && Math.abs(l10 - Math.round(l10)) < 1e-12)
    return `1e${Math.round(l10)}`;
  const l2 = Math.log2(n);
  if (n >= 1024 && Number.isInteger(l2)) return `2^${l2}`;
  return String(n);
}

export function fmtMs(ms: number | undefined): string {
  if (ms === undefined || !Number.isFinite(ms)) return "";
  const p = (v: number) => (v >= 100 ? v.toFixed(0) : v.toPrecision(3));
  if (ms < 1e-3) return `${p(ms * 1e6)} ns`;
  if (ms < 1) return `${p(ms * 1e3)} µs`;
  if (ms < 1e3) return `${p(ms)} ms`;
  return `${p(ms / 1e3)} s`;
}

function table(header: string[], rows: string[][], align?: string[]): string {
  const al = align ?? header.map((_, i) => (i === 0 ? "---" : "--:"));
  return [header, al, ...rows].map(r => `| ${r.join(" | ")} |`).join("\n");
}

function cell(e: Entry | undefined): string {
  if (!e) return "";
  if (e.status === "skipped") return "skip";
  if (e.status === "error") return "error";
  return fmtMs(e.median_ms);
}

export function renderMarkdown(r: BenchResult): string {
  const m = r.machine;
  const modes = r.modes.map(x => x.name);
  const byKey = new Map(
    r.entries.map(e => [`${e.case}|${e.mode}|${e.size}`, e])
  );
  const out: string[] = [];
  out.push(`# numbl benchmark results`);
  out.push("");
  out.push(
    `- Machine: ${m.cpu}, ${m.cores} logical cores, ${m.memGB} GB, ${m.os}`
  );
  out.push(
    `- numbl ${m.numblVersion} (commit ${m.commit}${m.dirty ? ", dirty" : ""}), Node ${m.node}, ${m.cc ?? "no cc"}`
  );
  out.push(
    `- Native addon: ${m.nativeAddon ? "loaded" : "not built"}; koffi: ${m.koffi ? "yes" : "no"}` +
      (Object.keys(m.env).length
        ? `; env: ${Object.entries(m.env)
            .map(([k, v]) => `${k}=${v}`)
            .join(" ")}`
        : "")
  );
  out.push(
    `- Run: ${r.created}, ${r.config.reps} reps, ${r.totalWallS} s total${r.config.quick ? " (--quick)" : ""}` +
      (m.loadavg
        ? `; load average ${m.loadavg[0].toFixed(1)} at start, ${m.loadavg[1].toFixed(1)} at end`
        : "")
  );
  out.push("");
  out.push(
    "Times are the median per-call time of the operation, measured inside numbl with tic/toc around a batch of calls (warm-up and JIT compilation excluded). `skip` means the size was skipped because the predicted call time exceeded the per-call limit; `error` means the run failed (see the JSON)."
  );
  out.push("");

  out.push(`## Process start-up`);
  out.push("");
  out.push(
    table(
      ["Mode", "Flags", "Start-up (trivial script)"],
      r.modes.map(x => [x.name, `\`${x.args}\``, fmtMs(x.startup_ms)])
    )
  );
  out.push("");

  out.push(`## Per-call time`);
  for (const c of r.cases) {
    const sizes = [
      ...new Set(r.entries.filter(e => e.case === c.name).map(e => e.size)),
    ].sort((a, b) => a - b);
    const caseModes = modes.filter(md =>
      r.entries.some(e => e.case === c.name && e.mode === md)
    );
    out.push("");
    out.push(`### ${c.name}: \`${c.description}\``);
    out.push("");
    out.push(
      table(
        [c.sizeLabel, ...caseModes],
        sizes.map(s => [
          fmtSize(s),
          ...caseModes.map(md => cell(byKey.get(`${c.name}|${md}|${s}`))),
        ])
      )
    );
  }
  out.push("");

  out.push(`## First-call overhead`);
  out.push("");
  out.push(
    "First call minus the steady-state per-call time, at the smallest size. This is mostly JIT compilation (including the C compile at `--opt 2` when the C-JIT cache is cold) and first-use warm-up."
  );
  out.push("");
  const rows: string[][] = [];
  for (const c of r.cases) {
    const row = [c.name];
    for (const md of modes) {
      const es = r.entries
        .filter(e => e.case === c.name && e.mode === md && e.status === "ok")
        .sort((a, b) => a.size - b.size);
      const e = es[0];
      row.push(e ? fmtMs(Math.max(0, e.first_ms! - e.median_ms!)) : "");
    }
    rows.push(row);
  }
  out.push(table(["Case", ...modes], rows));
  out.push("");

  const errors = r.entries.filter(e => e.status === "error");
  if (errors.length) {
    out.push(`## Errors`);
    out.push("");
    for (const e of errors)
      out.push(`- ${e.case} ${e.mode} n=${fmtSize(e.size)}: ${e.error}`);
    out.push("");
  }

  const mismatches: string[] = [];
  const groups = new Map<string, Entry[]>();
  for (const e of r.entries) {
    if (e.status !== "ok") continue;
    const k = `${e.case}|${e.size}`;
    groups.set(k, [...(groups.get(k) ?? []), e]);
  }
  for (const [k, es] of groups) {
    const ref = es[0].checksum!;
    for (const e of es.slice(1)) {
      const rel = Math.abs(e.checksum! - ref) / Math.max(Math.abs(ref), 1e-300);
      if (rel > 1e-6)
        mismatches.push(
          `- ${k.replace("|", " n=")}: ${es[0].mode} ${ref} vs ${e.mode} ${e.checksum} (rel. diff ${rel.toExponential(1)})`
        );
    }
  }
  if (mismatches.length) {
    out.push(`## Checksum mismatches across modes`);
    out.push("");
    out.push(...mismatches);
    out.push("");
  }

  if (r.diagnostics?.length) {
    // Group by case and message (without executor name and source
    // position), listing the modes that printed it.
    const grouped = new Map<string, Set<string>>();
    for (const d of r.diagnostics)
      for (const l of d.lines) {
        const msg = l
          .replace(/^\[jit-decline\] \S+( \[[^\]]*\])?: /, "JIT declined: ")
          .replace(/'double\[[^\]]*\]'/g, "'double[...]'");
        const k = `${d.case}\u0000${msg}`;
        grouped.set(k, (grouped.get(k) ?? new Set()).add(d.mode));
      }
    out.push(`## JIT diagnostics`);
    out.push("");
    out.push(
      "JIT declines and warnings printed while each kernel ran (`NUMBL_JIT_LOG=1`; the JSON has the full lines). A declined kernel runs on the interpreter in that mode."
    );
    out.push("");
    for (const [k, ms] of grouped) {
      const [c, msg] = k.split("\u0000");
      out.push(`- ${c} (${[...ms].join(", ")}): ${msg.replace(/`/g, "'")}`);
    }
    out.push("");
  }
  return out.join("\n");
}

export function compareResults(
  base: BenchResult,
  next: BenchResult,
  threshold: number
): { text: string; regressions: number } {
  const baseMap = new Map(
    base.entries.map(e => [`${e.case}|${e.mode}|${e.size}`, e])
  );
  const rows: { e: Entry; b: Entry; ratio: number }[] = [];
  for (const e of next.entries) {
    const b = baseMap.get(`${e.case}|${e.mode}|${e.size}`);
    if (!b || b.status !== "ok" || e.status !== "ok") continue;
    rows.push({ e, b, ratio: e.median_ms! / b.median_ms! });
  }
  const out: string[] = [];
  out.push(
    `Base: ${base.machine.commit} (${base.created}), new: ${next.machine.commit} (${next.created})`
  );
  if (base.machine.cpu !== next.machine.cpu)
    out.push(
      `Warning: different machines (${base.machine.cpu} vs ${next.machine.cpu})`
    );
  const geo = (xs: number[]) =>
    xs.length
      ? Math.exp(xs.reduce((s, x) => s + Math.log(x), 0) / xs.length)
      : NaN;
  out.push(
    `${rows.length} matching entries; geometric-mean time ratio new/base = ${geo(rows.map(r => r.ratio)).toFixed(3)}`
  );
  const modes = [...new Set(rows.map(r => r.e.mode))];
  out.push(
    "Per mode: " +
      modes
        .map(
          md =>
            `${md} ${geo(rows.filter(r => r.e.mode === md).map(r => r.ratio)).toFixed(3)}`
        )
        .join(", ")
  );
  const fmtRow = (r: (typeof rows)[number]) => [
    r.e.case,
    r.e.mode,
    fmtSize(r.e.size),
    fmtMs(r.b.median_ms),
    fmtMs(r.e.median_ms),
    `${r.ratio.toFixed(2)}x`,
  ];
  const header = ["Case", "Mode", "Size", "Base", "New", "New/base"];
  const reg = rows
    .filter(r => r.ratio > 1 + threshold)
    .sort((a, b) => b.ratio - a.ratio);
  const imp = rows
    .filter(r => r.ratio < 1 / (1 + threshold))
    .sort((a, b) => a.ratio - b.ratio);
  out.push("");
  out.push(
    `## Regressions (slower by more than ${(threshold * 100).toFixed(0)}%): ${reg.length}`
  );
  if (reg.length) out.push("", table(header, reg.map(fmtRow)));
  out.push("");
  out.push(`## Improvements: ${imp.length}`);
  if (imp.length) out.push("", table(header, imp.map(fmtRow)));
  const missing = base.entries.filter(
    b =>
      b.status === "ok" &&
      next.entries.some(
        e =>
          e.case === b.case &&
          e.mode === b.mode &&
          e.size === b.size &&
          e.status === "error"
      )
  );
  if (missing.length) {
    out.push("");
    out.push(`## Newly failing: ${missing.length}`);
    for (const b of missing)
      out.push(`- ${b.case} ${b.mode} n=${fmtSize(b.size)}`);
  }
  return { text: out.join("\n"), regressions: reg.length + missing.length };
}
