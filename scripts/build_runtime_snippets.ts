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
 * It also emits the snippet table (`SNIPPET_DEFS`, `SNIPPET_ALIASES`)
 * that `codegen/runtime.ts` activates from. Each `.h` declares its entry
 * in a required first-line header, stripped from the emitted C:
 *
 *   /* @snippet name=jit_disp_double deps=jit_format_double,jit_host_output *\/
 *
 *   name=     the name emitted code activates (`useRuntime("jit_...")`)
 *   deps=     snippets the C body needs, activated (emitted) before it
 *   aliases=  other names that activate the same snippet
 *
 * A `.js` twin's dependencies are not declared: the `.h` twins of the
 * files it imports are appended to `deps`.
 *
 * .js snippet contract:
 *   - A valid ES module (so builtin defs can
 *     `import { foo } from "./snippets.gen.js"`).
 *   - Cross-snippet imports must use relative paths to another `.js`
 *     under the runtime tree (`./peer.js`, `../topic/peer.js`, …).
 *     Node resolves these via real ESM; the codegen path
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
import { format, resolveConfig } from "prettier";

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
// The C runtime snippets (`C_SNIPPETS`) are emitted into a SEPARATE file so
// they can be excluded from the browser worker bundle: only the Node C-JIT
// bootstrap imports this module (registering the table via `setCSnippets`),
// while the browser-reachable codegen graph never references it.
const cOutFile = join(runtimeDir, "snippets.c.gen.ts");

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

/** Snippet basenames (with extension) that are intentionally JS-only — a
 *  `.js` runtime helper with no C-JIT counterpart. Empty today: every `.js`
 *  snippet currently has a behavior-matching `.h` twin, and the pair must stay
 *  in lockstep (a `.js` that drifts from its `.h` is exactly the bug class this
 *  guard defends against). If a genuinely JS-only helper is ever needed, add
 *  its basename here with a comment explaining why C-JIT doesn't need it. */
const JS_ONLY_SNIPPETS = new Set<string>([]);

/** Every `.js` snippet must have a sibling `.h` of the same stem (so the
 *  JS-JIT and C-JIT runtimes stay paired), unless explicitly allowlisted as
 *  JS-only. The reverse is NOT required: many `.h` snippets cover C-ABI
 *  concerns (alloc/free, string/char structs, longjmp guards) that JS handles
 *  natively and so have no `.js` twin. */
function assertJsSnippetsHaveHTwin(
  hPaths: ReadonlyArray<string>,
  jsPaths: ReadonlyArray<string>
): void {
  const hStems = new Set(hPaths.map(p => basename(p).replace(/\.h$/, "")));
  const orphans: string[] = [];
  for (const p of jsPaths) {
    const b = basename(p);
    if (JS_ONLY_SNIPPETS.has(b)) continue;
    if (!hStems.has(b.replace(/\.js$/, ""))) orphans.push(p);
  }
  if (orphans.length > 0) {
    throw new Error(
      `runtime snippet pairing: the following .js snippet(s) have no sibling ` +
        `.h twin: ${orphans.join(", ")}. The JS-JIT (.js) and C-JIT (.h) ` +
        `runtimes must stay paired; add the matching .h, or (if the helper is ` +
        `genuinely JS-only) add its basename to JS_ONLY_SNIPPETS in ` +
        `scripts/build_runtime_snippets.ts with a justifying comment.`
    );
  }
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

interface SnippetHeader {
  name: string;
  deps: string[];
  aliases: string[];
}

const HEADER_RE = /^\/\* @snippet ([^\n]*?) \*\/\r?\n/;

/** Parse and strip the `@snippet` header (the required first line) of a
 *  `.h` snippet. */
function parseHeader(
  src: string,
  file: string
): { header: SnippetHeader; body: string } {
  const m = src.match(HEADER_RE);
  if (!m) {
    throw new Error(
      `${file}: missing snippet header; the first line must be ` +
        "`/* @snippet name=jit_<name> [deps=a,b] [aliases=c,d] */`"
    );
  }
  const header: SnippetHeader = { name: "", deps: [], aliases: [] };
  for (const field of m[1].trim().split(/\s+/)) {
    const kv = field.match(
      /^(name|deps|aliases)=([A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*)$/
    );
    if (!kv) throw new Error(`${file}: bad snippet header field '${field}'`);
    if (kv[1] === "name") header.name = kv[2];
    else header[kv[1] as "deps" | "aliases"] = kv[2].split(",");
  }
  if (!/^[A-Za-z0-9_]+$/.test(header.name)) {
    throw new Error(`${file}: snippet header needs one name=`);
  }
  return { header, body: src.slice(m[0].length) };
}

/** Basenames a `.js` snippet imports, deduplicated and sorted. */
function jsImportBasenames(src: string): string[] {
  const imports = new Set<string>();
  for (const m of src.matchAll(/^\s*import\s+.*?from\s+["']([^"']+)["']/gm)) {
    imports.add(basename(m[1]));
  }
  return Array.from(imports).sort();
}

function generate(): { js: string; c: string } {
  const allPaths = walkRuntime(runtimeDir);

  const hPaths = allPaths.filter(p => p.endsWith(".h"));
  const jsPaths = allPaths.filter(p => p.endsWith(".js"));
  // `.d.ts` files (ambient declarations) are typecheck-only and not
  // inlined; their presence on disk is enough.
  assertUniqueBasenames([...hPaths, ...jsPaths]);
  assertJsSnippetsHaveHTwin(hPaths, jsPaths);

  // Every `.h` carries a header naming its snippet; the C body is the
  // file without that line.
  const parsed = hPaths.map(p => {
    const { header, body } = parseHeader(
      readFileSync(join(runtimeDir, p), "utf8"),
      p
    );
    return { file: basename(p), header, body };
  });
  const cEntries = parsed.map(
    ({ file, body }) => `  ${JSON.stringify(file)}: ${JSON.stringify(body)},`
  );

  // Names and aliases must be unique; deps must name a declared snippet.
  const fileToName = new Map<string, string>();
  const known = new Set<string>();
  const declare = (name: string, file: string): void => {
    if (known.has(name)) {
      throw new Error(`${file}: snippet name '${name}' is declared twice`);
    }
    known.add(name);
  };
  for (const { file, header } of parsed) {
    declare(header.name, file);
    for (const a of header.aliases) declare(a, file);
    fileToName.set(file, header.name);
  }

  const jsSources = new Map<string, string>();
  for (const p of jsPaths) {
    const body = readFileSync(join(runtimeDir, p), "utf8");
    assertRelativeJsImport(body, p);
    jsSources.set(basename(p), body);
  }

  // The snippet table. A snippet's `deps` are the header's `deps`
  // followed by the snippets its paired `.js` imports (each import's `.h`
  // twin), in the order the activator pulls them in.
  const tableEntries: string[] = [];
  const aliasEntries: string[] = [];
  for (const { file, header } of parsed) {
    for (const d of header.deps) {
      if (!known.has(d)) {
        throw new Error(`${file}: unknown snippet dependency '${d}'`);
      }
    }
    const deps = [...header.deps];
    const jsSrc = jsSources.get(file.replace(/\.h$/, ".js"));
    for (const imp of jsSrc === undefined ? [] : jsImportBasenames(jsSrc)) {
      const depName = fileToName.get(imp.replace(/\.js$/, ".h"));
      if (depName === undefined) {
        throw new Error(`${file}: its .js imports '${imp}', which has no .h`);
      }
      if (!deps.includes(depName)) deps.push(depName);
    }
    tableEntries.push(
      `  { name: ${JSON.stringify(header.name)}, ` +
        `file: ${JSON.stringify(file)}, deps: ${JSON.stringify(deps)} },`
    );
    for (const a of header.aliases) {
      aliasEntries.push(
        `  ${JSON.stringify(a)}: ${JSON.stringify(header.name)},`
      );
    }
  }

  const jsEntries = jsPaths.map(p => {
    const stripped = stripJsExportsAndImports(jsSources.get(basename(p))!);
    return `  ${JSON.stringify(basename(p))}: ${JSON.stringify(stripped)},`;
  });

  // Re-export every named export from each .js snippet via its
  // relative path so builtin defs can `import { foo } from
  // "./snippets.gen.js"` for compile-time folding — the same source
  // of truth the emitter inlines.
  const reExports = jsPaths
    .map(p => `export * from ${JSON.stringify("./" + p)};`)
    .join("\n");

  const js = [
    "// DO NOT EDIT — generated by scripts/build_runtime_snippets.ts.",
    "// Re-run `npm run build:snippets` after editing any runtime/*.h",
    "// or runtime/*.js file.",
    "",
    "// Re-exports of every named export from each .js snippet, so",
    "// builtin defs can fold constants with the same code the emitters",
    "// inline.",
    reExports,
    "",
    "/** JS snippet bodies (with `export` keywords stripped), keyed by",
    " *  basename (e.g. `disp_double.js`). Inlined into emitted JS. The",
    " *  functions may reference host hooks (`$write`, …) as free",
    " *  variables that resolve to `globalThis.<name>` at call time. */",
    "export const JS_SNIPPETS: Record<string, string> = {",
    ...jsEntries,
    "};",
    "",
    "/** One runtime snippet: the name emitted code activates, the `.h`",
    " *  file it is loaded from (a `.js` twin shares its stem), and the",
    " *  snippets activated before it: the header's `deps`, then the `.h`",
    " *  twins of the `.js` imports. */",
    "export interface SnippetDef {",
    "  name: string;",
    "  file: string;",
    "  deps: string[];",
    "}",
    "",
    "/** Every runtime snippet, from the `@snippet` header of each `.h`. */",
    "export const SNIPPET_DEFS: SnippetDef[] = [",
    ...tableEntries,
    "];",
    "",
    "/** Extra activation names (header `aliases=`) → snippet name. */",
    "export const SNIPPET_ALIASES: Record<string, string> = {",
    ...aliasEntries,
    "};",
    "",
  ].join("\n");

  // C snippets live in their own module so the browser worker bundle can
  // exclude them: only the Node C-JIT bootstrap imports this file and feeds
  // the table to the codegen runtime via `setCSnippets`. See
  // `codegen/runtime.ts` and `executors/jit/compileC.node.ts`.
  const c = [
    "// DO NOT EDIT — generated by scripts/build_runtime_snippets.ts.",
    "// Re-run `npm run build:snippets` after editing any runtime/*.h",
    "// or runtime/*.js file.",
    "",
    "/** C snippet bodies, keyed by basename (e.g. `disp_double.h`).",
    " *  C-JIT only (`--opt 2`, Node). Registered at bootstrap via",
    " *  `setCSnippets` so the browser bundle never pulls it in. */",
    "export const C_SNIPPETS: Record<string, string> = {",
    ...cEntries,
    "};",
    "",
  ].join("\n");

  return { js, c };
}

// Format the generated source with the project's Prettier config so the
// written file is byte-identical to what the pre-commit hook would produce.
// Without this, `--check` compares the raw generator output (long single-line
// entries) against the committed, Prettier-wrapped file and would always
// report drift — making the check unusable in CI.
const { js: rawJs, c: rawC } = generate();
const prettierConfig = await resolveConfig(outFile);
const generatedJs = await format(rawJs, {
  ...prettierConfig,
  parser: "typescript",
  filepath: outFile,
});
const generatedC = await format(rawC, {
  ...prettierConfig,
  parser: "typescript",
  filepath: cOutFile,
});
const check = process.argv.includes("--check");

function checkFile(file: string, generated: string): boolean {
  let current = "";
  try {
    current = readFileSync(file, "utf8");
  } catch {
    /* file missing counts as drift */
  }
  return current === generated;
}

if (check) {
  const ok = checkFile(outFile, generatedJs) && checkFile(cOutFile, generatedC);
  if (!ok) {
    console.error(
      "snippets.gen.ts / snippets.c.gen.ts is out of date. Run " +
        "`npm run build:snippets` (did you edit a runtime/*.h or " +
        "runtime/*.js without regenerating?)."
    );
    process.exit(1);
  }
  process.exit(0);
}

writeFileSync(outFile, generatedJs);
console.log(`wrote ${outFile}`);
writeFileSync(cOutFile, generatedC);
console.log(`wrote ${cOutFile}`);
