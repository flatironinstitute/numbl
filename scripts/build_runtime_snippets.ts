#!/usr/bin/env tsx
/**
 * Generate src/builtins/runtime/snippets.gen.ts by inlining every .h
 * and .js file under src/builtins/runtime/ (recursing into the
 * topic subdirectories) as a string literal. The codegen runtime
 * module imports from the generated file instead of reading the
 * source files at runtime, which lets the translator bundle in the
 * browser.
 *
 * The `C_SNIPPETS` and `JS_SNIPPETS` maps are keyed by basename
 * (`disp_double.h`, `disp_double.js`) — the on-disk subdirectory
 * structure is purely for organization. Basenames are globally
 * unique across the runtime tree (enforced here).
 *
 * .js snippet contract:
 *   - A valid standalone ES module (so the interpreter side can
 *     `import { foo } from "./snippets.gen.js"`).
 *   - Cross-snippet imports must use relative paths to another `.js`
 *     under the runtime tree (`./peer.js`, `../topic/peer.js`, …).
 *     The interpreter resolves these via real ESM; the codegen path
 *     strips them and inlines all snippets together.
 *   - References host hooks (e.g. `$write`) as free variables that
 *     resolve to `globalThis.<name>` at call time.
 *   - Top-level names must not collide across snippet files.
 *
 *   npx tsx scripts/build_runtime_snippets.ts          # write the file
 *   npx tsx scripts/build_runtime_snippets.ts --check  # exit 1 if drifted
 */
import { readFileSync, readdirSync, writeFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve, relative, basename } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const runtimeDir = resolve(
  here,
  "..",
  "src",
  "numbl-core",
  "jit",
  "builtins",
  "runtime"
);
const outFile = join(runtimeDir, "snippets.gen.ts");

function stripJsExportsAndImports(src: string): string {
  // Remove `export ` from top-level function/const/let/class declarations.
  // The inlined codegen text needs plain declarations so the surrounding
  // emitted module can reference them by bare name.
  let out = src.replace(/^export\s+(function|const|let|class)\b/gm, "$1");
  // Drop `import` statements (single-line and multi-line braced
  // forms). The codegen path inlines every activated snippet into one
  // module, so cross-snippet calls resolve via module scope and the
  // imports become redundant. The non-greedy `[\s\S]*?` matches across
  // newlines for braced multi-line imports.
  out = out.replace(
    /^\s*import\s+[\s\S]*?from\s+["'][^"']+["'];?[ \t]*$/gm,
    ""
  );
  return out;
}

function assertRelativeJsImport(src: string, file: string): void {
  const matches = src.matchAll(/^\s*import\s+.*?from\s+["']([^"']+)["']/gm);
  for (const m of matches) {
    const spec = m[1];
    // Accept any relative path resolving to another `.js` under the
    // runtime tree; reject bare-package or absolute imports.
    if (!/^\.{1,2}\/[A-Za-z0-9_/]+\.js$/.test(spec)) {
      throw new Error(
        `${file}: snippet 'import' must reference another runtime .js ` +
          `via a relative path (e.g. "./peer.js" or "../topic/peer.js"); ` +
          `got "${spec}".`
      );
    }
  }
}

/** Walk runtime/ recursively. Skips `snippets.gen.ts` (the generated
 *  output) but otherwise picks up every `.h`, `.js`, and `.d.ts`
 *  under any subdirectory. Returns each entry's path relative to
 *  `runtimeDir` (e.g. `tensor/tensor_alloc.h`) so the build can keep
 *  basename keys in the map but still re-export via accurate
 *  relative-path strings. */
function walkRuntime(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir).sort()) {
    const p = join(dir, entry);
    const st = statSync(p);
    if (st.isDirectory()) {
      for (const child of walkRuntime(p)) {
        out.push(child);
      }
    } else if (st.isFile()) {
      const rel = relative(runtimeDir, p).replace(/\\/g, "/");
      if (rel === "snippets.gen.ts") continue;
      out.push(rel);
    }
  }
  return out;
}

function assertUniqueBasenames(paths: ReadonlyArray<string>): void {
  const seen = new Map<string, string>();
  for (const p of paths) {
    const b = basename(p);
    const prev = seen.get(b);
    if (prev !== undefined) {
      throw new Error(
        `runtime snippet basename collision: '${b}' appears at both ` +
          `'${prev}' and '${p}'. Basenames must be globally unique across ` +
          `the runtime tree because the snippet registry keys by basename.`
      );
    }
    seen.set(b, p);
  }
}

function generate(): string {
  const allPaths = walkRuntime(runtimeDir);

  const hPaths = allPaths.filter(p => p.endsWith(".h"));
  const jsPaths = allPaths.filter(p => p.endsWith(".js"));
  // `.d.ts` files (ambient declarations) are typecheck-only and not
  // inlined; their presence on disk is enough.
  assertUniqueBasenames([...hPaths, ...jsPaths]);

  const cEntries = hPaths.map(p => {
    const body = readFileSync(join(runtimeDir, p), "utf8");
    const key = basename(p);
    return `  ${JSON.stringify(key)}: ${JSON.stringify(body)},`;
  });

  const jsEntries = jsPaths.map(p => {
    const body = readFileSync(join(runtimeDir, p), "utf8");
    assertRelativeJsImport(body, p);
    const stripped = stripJsExportsAndImports(body);
    const key = basename(p);
    return `  ${JSON.stringify(key)}: ${JSON.stringify(stripped)},`;
  });

  // Per-file list of imported snippet basenames (e.g.
  // `tensor_elemwise_real.js` → `["tensor_alloc_nd.js"]`). The codegen
  // path turns each entry into the corresponding registered snippet
  // name (`mtoc2_<basename without ext>`) so activation pulls JS deps
  // in alongside the C deps declared in the registry.
  const jsImportEntries = jsPaths.map(p => {
    const body = readFileSync(join(runtimeDir, p), "utf8");
    const imports = new Set<string>();
    for (const m of body.matchAll(
      /^\s*import\s+.*?from\s+["']([^"']+)["']/gm
    )) {
      imports.add(basename(m[1]));
    }
    const key = basename(p);
    const list = Array.from(imports)
      .sort()
      .map(b => JSON.stringify(b))
      .join(", ");
    return `  ${JSON.stringify(key)}: [${list}],`;
  });

  // Re-export every named export from each .js snippet via its
  // relative path so the interpreter (Phase 3) can `import { foo }
  // from "./snippets.gen.js"` and call helpers directly — the same
  // source of truth the emitter inlines.
  const reExports = jsPaths
    .map(p => `export * from ${JSON.stringify("./" + p)};`)
    .join("\n");

  return [
    "// DO NOT EDIT — generated by scripts/build_runtime_snippets.ts.",
    "// Re-run `npm run build:snippets` after editing any runtime/*.h",
    "// or runtime/*.js file.",
    "",
    "// Re-exports of every named export from each .js snippet, so the",
    "// interpreter / `Builtin.call` definitions can import helpers",
    "// directly and run the same code the emitters inline.",
    reExports,
    "",
    "/** C snippet bodies, keyed by basename (e.g. `disp_double.h`). */",
    "export const C_SNIPPETS: Record<string, string> = {",
    ...cEntries,
    "};",
    "",
    "/** JS snippet bodies (with `export` keywords stripped), keyed by",
    " *  basename (e.g. `disp_double.js`). Inlined into emitted JS. The",
    " *  functions may reference host hooks (`$write`, …) as free",
    " *  variables that resolve to `globalThis.<name>` at call time. */",
    "export const JS_SNIPPETS: Record<string, string> = {",
    ...jsEntries,
    "};",
    "",
    "/** Per-JS-snippet list of basenames it imports from sibling JS",
    " *  snippets (e.g. `tensor_elemwise_real.js` → ",
    " *  `['tensor_alloc_nd.js']`). The codegen path converts each entry",
    " *  to a registered snippet name (`mtoc2_<basename without .js>`) so",
    " *  cross-snippet JS deps activate alongside the explicit C deps. */",
    "export const JS_IMPORTS: Record<string, string[]> = {",
    ...jsImportEntries,
    "};",
    "",
  ].join("\n");
}

const generated = generate();
const check = process.argv.includes("--check");

if (check) {
  let current = "";
  try {
    current = readFileSync(outFile, "utf8");
  } catch {
    /* file missing counts as drift */
  }
  if (current !== generated) {
    console.error(
      "snippets.gen.ts is out of date. Run `npm run build:snippets`."
    );
    process.exit(1);
  }
  process.exit(0);
}

writeFileSync(outFile, generated);
console.log(`wrote ${outFile}`);
