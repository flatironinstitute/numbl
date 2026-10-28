#!/usr/bin/env tsx
/**
 * Count circular imports among the modules of src/numbl-core and fail if the
 * count rises above MAX_CYCLES. Lower MAX_CYCLES when you remove cycles.
 *
 * The count uses the same DFS as `npx madge --circular --extensions ts
 * src/numbl-core` (one cycle per back edge, type-only imports included), so
 * madge can be used to list the cycles themselves.
 *
 *   npx tsx scripts/check_import_cycles.ts           # check against the pin
 *   npx tsx scripts/check_import_cycles.ts --list    # also print each cycle
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import ts from "typescript";

const MAX_CYCLES = 62;

const ROOT = resolve(import.meta.dirname, "../src/numbl-core");

function listFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) listFiles(p, out);
    else if (p.endsWith(".ts") && !p.endsWith(".d.ts")) out.push(p);
  }
  return out;
}

function resolveImport(from: string, spec: string): string | null {
  if (!spec.startsWith(".")) return null;
  const base = resolve(dirname(from), spec);
  const candidates = [
    base.replace(/\.js$/, ".ts"),
    base + ".ts",
    join(base, "index.ts"),
  ];
  const hit = candidates.find(c => c.startsWith(ROOT) && existsSync(c));
  return hit && statSync(hit).isFile() ? hit : null;
}

/** Every module `file` imports: import and `export ... from` declarations,
 *  dynamic `import()` calls and `import("...")` types, as madge counts them. */
function importsOf(file: string): string[] {
  const sf = ts.createSourceFile(
    file,
    readFileSync(file, "utf8"),
    ts.ScriptTarget.Latest
  );
  const deps = new Set<string>();
  const add = (spec: ts.Node | undefined) => {
    if (!spec || !ts.isStringLiteralLike(spec)) return;
    const dep = resolveImport(file, spec.text);
    if (dep) deps.add(dep);
  };
  const walk = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      add(node.moduleSpecifier);
    } else if (ts.isImportTypeNode(node)) {
      const arg = node.argument;
      if (ts.isLiteralTypeNode(arg)) add(arg.literal);
    } else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword
    ) {
      add(node.arguments[0]);
    }
    ts.forEachChild(node, walk);
  };
  walk(sf);
  return [...deps];
}

// Like madge, visit modules and their imports in sorted path order.
const graph = new Map<string, string[]>();
for (const f of listFiles(ROOT).sort()) graph.set(f, importsOf(f).sort());

// madge's cycle search: DFS, recording the open path on each back edge.
const cycles: string[][] = [];
const resolved = new Set<string>();
const onPath: string[] = [];
function visit(id: string): void {
  onPath.push(id);
  for (const dep of graph.get(id) ?? []) {
    if (resolved.has(dep)) continue;
    const i = onPath.indexOf(dep);
    if (i >= 0) {
      cycles.push(onPath.slice(i));
      continue;
    }
    visit(dep);
  }
  onPath.pop();
  resolved.add(id);
}
for (const id of graph.keys()) if (!resolved.has(id)) visit(id);

if (process.argv.includes("--list")) {
  cycles.forEach((c, i) =>
    console.log(`${i + 1}) ${c.map(f => relative(ROOT, f)).join(" > ")}`)
  );
}

const n = cycles.length;
if (n > MAX_CYCLES) {
  console.error(
    `src/numbl-core has ${n} circular imports, above the pinned maximum of ${MAX_CYCLES}.\n` +
      `Run \`npx tsx scripts/check_import_cycles.ts --list\` to see them, and avoid ` +
      `importing a module that (transitively) imports yours; barrel files ` +
      `such as runtime/index.ts are a common cause.`
  );
  process.exit(1);
}
console.log(`src/numbl-core: ${n} circular imports (max ${MAX_CYCLES}).`);
if (n < MAX_CYCLES) {
  console.log(
    `Lower MAX_CYCLES in scripts/check_import_cycles.ts to ${n} to lock in the improvement.`
  );
}
