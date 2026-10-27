/**
 * `numbl build-site`: bundle a project directory plus the prebuilt browser IDE
 * (dist-site-viewer/) into a static site. See docs/developer_reference/web-app.md.
 */

import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "fs";
import { basename, join, relative, resolve, sep } from "path";
import { createHash } from "crypto";
import { zipSync } from "fflate";
import { parseCommandArgs } from "./cli-args.js";

const SITE_DEFAULT_IGNORE_DIRS = new Set([".git", ".github", "node_modules"]);
const SITE_MAX_FILE_BYTES = 25 * 1024 * 1024;

/** Coding-agent instruction files (CLAUDE.md, AGENTS.md, AGENT*.md, GEMINI.md,
 *  …). These are dev tooling, not project content, so build-site never bundles
 *  them into the rendered site (matched by basename, at any depth). */
function isAgentInstructionFile(name: string): boolean {
  return name === "GEMINI.md" || /^(CLAUDE|AGENT)[^/]*\.md$/i.test(name);
}

/** Read .numblignore (gitignore-ish: blank/`#` lines skipped). */
function readNumblIgnore(projectDir: string): string[] {
  const p = join(projectDir, ".numblignore");
  if (!existsSync(p)) return [];
  return readFileSync(p, "utf-8")
    .split("\n")
    .map(l => l.trim())
    .filter(l => l && !l.startsWith("#"));
}

/** Compile a gitignore-ish glob (`*` = within a segment, `**` = any) to regex. */
function globToRegExp(glob: string): RegExp {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const expanded = escaped.replace(/\*\*|\*/g, m =>
    m === "**" ? ".*" : "[^/]*"
  );
  return new RegExp("^" + expanded + "$");
}

/** Walk a project dir collecting files to bundle, applying ignore rules. */
function collectSiteFiles(
  projectDir: string,
  outDirAbs: string,
  ignore: string[]
): { path: string; data: Uint8Array }[] {
  const out: { path: string; data: Uint8Array }[] = [];
  const matchesIgnore = (rel: string): boolean => {
    const base = rel.split("/").pop() ?? rel;
    for (const pat of ignore) {
      const norm = pat.replace(/\/+$/, "");
      if (!norm) continue;
      // Exact path or directory-prefix match.
      if (rel === norm || rel.startsWith(norm + "/")) return true;
      if (pat.includes("*")) {
        const re = globToRegExp(norm);
        if (re.test(rel) || re.test(base)) return true;
      } else if (base === norm) {
        return true;
      }
    }
    return false;
  };
  const walk = (dir: string) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const e of entries) {
      const full = join(dir, e.name);
      if (resolve(full) === outDirAbs) continue; // never bundle the output dir
      const rel = relative(projectDir, full).split(sep).join("/");
      if (matchesIgnore(rel)) continue;
      if (e.isDirectory()) {
        if (SITE_DEFAULT_IGNORE_DIRS.has(e.name)) continue;
        walk(full);
      } else if (e.isFile()) {
        if (e.name === ".numblignore" || e.name === ".DS_Store") continue;
        if (isAgentInstructionFile(e.name)) continue;
        const st = statSync(full);
        if (st.size > SITE_MAX_FILE_BYTES) {
          console.error(
            `warning: skipping ${rel} (${(st.size / 1048576).toFixed(1)} MB exceeds 25 MB per-file cap)`
          );
          continue;
        }
        out.push({ path: rel, data: new Uint8Array(readFileSync(full)) });
      }
    }
  };
  walk(projectDir);
  return out;
}

export async function cmdBuildSite(
  args: string[],
  packageDir: string
): Promise<void> {
  const { values, positionals } = parseCommandArgs(args, {
    out: { type: "string" },
    base: { type: "string" },
    title: { type: "string" },
    entry: { type: "string" },
    "repo-url": { type: "string" },
    "max-initial-output-panel-height": { type: "string" },
  });
  const { base, title, entry } = values;
  const repoUrl = values["repo-url"];
  let maxInitialOutputPanelHeight: number | undefined;
  if (values["max-initial-output-panel-height"] !== undefined) {
    const n = Number(values["max-initial-output-panel-height"]);
    if (!Number.isFinite(n) || n <= 0) {
      console.error(
        "Error: --max-initial-output-panel-height must be a positive number"
      );
      process.exit(1);
    }
    maxInitialOutputPanelHeight = n;
  }
  if (positionals.length > 1) {
    console.error("Error: numbl build-site accepts at most one directory");
    process.exit(1);
  }
  const projectDir = positionals[0] ?? ".";
  const outDir = values.out ?? "dist";

  const projectDirAbs = resolve(process.cwd(), projectDir);
  const outDirAbs = resolve(process.cwd(), outDir);
  const siteViewerDir = join(packageDir, "dist-site-viewer");

  if (!existsSync(join(siteViewerDir, "index.html"))) {
    console.error(`Error: site viewer bundle not found at ${siteViewerDir}.`);
    console.error(
      "If running from source, build it first: npm run build:site-viewer"
    );
    process.exit(1);
  }
  if (!existsSync(projectDirAbs) || !statSync(projectDirAbs).isDirectory()) {
    console.error(`Error: project directory not found: ${projectDirAbs}`);
    process.exit(1);
  }

  // Normalize the base to /…/ form when provided.
  let normBase: string | undefined;
  if (base !== undefined) {
    normBase = base.startsWith("/") ? base : "/" + base;
    if (!normBase.endsWith("/")) normBase += "/";
  }

  console.log(`Building numbl site:`);
  console.log(`  project: ${projectDirAbs}`);
  console.log(`  viewer:  ${siteViewerDir}`);
  console.log(`  output:  ${outDirAbs}`);
  if (normBase) console.log(`  base:    ${normBase}`);

  // 1. Reset output and copy the prebuilt viewer bundle.
  rmSync(outDirAbs, { recursive: true, force: true });
  mkdirSync(outDirAbs, { recursive: true });
  cpSync(siteViewerDir, outDirAbs, { recursive: true });

  // 2. Collect project files (binary-safe), honoring ignore rules.
  const ignore = readNumblIgnore(projectDirAbs);
  const collected = collectSiteFiles(projectDirAbs, outDirAbs, ignore);
  if (collected.length === 0) {
    console.error("warning: no project files found to bundle");
  }

  // 3. Manifest = existing numbl-project.json (if any) merged with flags.
  let manifest: {
    title?: string;
    entry?: string;
    repository?: string;
    maxInitialOutputPanelHeight?: number;
  } = {};
  const existing = collected.find(f => f.path === "numbl-project.json");
  if (existing) {
    try {
      manifest = JSON.parse(new TextDecoder().decode(existing.data));
    } catch {
      // ignore malformed manifest in the project tree
    }
  }
  if (title !== undefined) manifest.title = title;
  if (entry !== undefined) manifest.entry = entry;
  if (repoUrl !== undefined) manifest.repository = repoUrl;
  if (maxInitialOutputPanelHeight !== undefined)
    manifest.maxInitialOutputPanelHeight = maxInitialOutputPanelHeight;
  if (!manifest.title) manifest.title = basename(projectDirAbs);

  // 4. Build project.zip (manifest replaces any tree copy of itself).
  const zipEntries: Record<string, Uint8Array> = {};
  for (const f of collected) {
    if (f.path === "numbl-project.json") continue;
    zipEntries[f.path] = f.data;
  }
  zipEntries["numbl-project.json"] = new TextEncoder().encode(
    JSON.stringify(manifest, null, 2)
  );
  const zip = zipSync(zipEntries, { level: 6 });
  writeFileSync(join(outDirAbs, "project.zip"), zip);
  const totalMb = zip.length / 1048576;
  console.log(
    `  bundled ${Object.keys(zipEntries).length} files → project.zip (${totalMb.toFixed(2)} MB)`
  );
  if (totalMb > 50) {
    console.error(
      "warning: project bundle exceeds 50 MB; GitHub Pages may be slow or reject it"
    );
  }

  // A build id derived from the bundle contents. Injected into the page and
  // used as a cache-busting query on the project.zip fetch: a redeploy changes
  // the bundle (and thus the id) so clients fetch the new zip, while reloads of
  // the same build keep the same id and stay cached.
  const buildId = createHash("sha256").update(zip).digest("hex").slice(0, 16);

  // 5. Inject runtime globals into the page: the build id (always) and the
  //    deploy base (when set, so project.zip resolves under a subpath).
  const indexPath = join(outDirAbs, "index.html");
  let injectJs = `window.__NUMBL_BUILD_ID__=${JSON.stringify(buildId)};`;
  if (normBase) {
    injectJs += `window.__NUMBL_BASE__=${JSON.stringify(normBase)};`;
  }
  const inject = `<script>${injectJs}</script>`;
  let indexHtml = readFileSync(indexPath, "utf-8");
  indexHtml = indexHtml.includes("<!-- numbl:base -->")
    ? indexHtml.replace("<!-- numbl:base -->", inject)
    : indexHtml.replace("</head>", `  ${inject}\n  </head>`);
  // Set the document <title> (browser tab name) to the project title so it
  // reflects the project rather than the generic placeholder. SiteApp also
  // keeps document.title in sync at runtime; this avoids a flash on first paint.
  const escapeHtml = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  indexHtml = indexHtml.replace(
    /<title>.*?<\/title>/,
    `<title>${escapeHtml(manifest.title)}</title>`
  );
  writeFileSync(indexPath, indexHtml);

  // 6. A 404 that bounces to the deploy root, plus .nojekyll for GitHub Pages.
  const redirectTarget = normBase ?? "/";
  writeFileSync(
    join(outDirAbs, "404.html"),
    `<!doctype html><html><head><meta charset="utf-8">` +
      `<script>location.replace(${JSON.stringify(redirectTarget)});</script>` +
      `</head><body></body></html>\n`
  );
  writeFileSync(join(outDirAbs, ".nojekyll"), "");

  console.log(
    "Done. Deploy the output directory to a static host (e.g. GitHub Pages)."
  );
}
