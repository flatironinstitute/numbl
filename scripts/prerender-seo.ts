/**
 * Prerender static HTML for the public, indexable routes.
 *
 * numbl.org is a client-side SPA on GitHub Pages, so without this every route
 * except "/" is served by the 404.html SPA-redirect shell (HTTP 404, no real
 * content, shared <title>). This script clones the built dist/index.html for
 * each SEO route and rewrites its <head> metadata (title, description,
 * canonical, Open Graph / Twitter, JSON-LD). For docs and home it also injects
 * crawlable static content into #root — React clears and re-renders #root on
 * mount, so this is only ever seen by crawlers and during first paint.
 *
 * Also emits sitemap.xml. Runs after `vite build`; see package.json.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve, join } from "node:path";
import { marked } from "marked";
import { docsManifest } from "../src/docs/manifest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DIST = join(ROOT, "dist");
const DOCS_DIR = join(ROOT, "src", "docs");

const SITE = "https://numbl.org";
const HOME_TITLE =
  "numbl — MATLAB-Compatible Numerical Computing in Your Browser";
const HOME_DESC =
  "Free, open-source numerical computing environment compatible with MATLAB syntax. 400+ built-in functions, 2-D and 3-D plotting, and a browser IDE with no installation — plus a Node.js CLI.";

interface Route {
  /** URL path, e.g. "/" or "/docs/language". No trailing slash except root. */
  path: string;
  title: string;
  description: string;
  /** Static HTML injected into #root for crawlers / first paint. */
  bodyHtml: string;
  /** Extra JSON-LD (replaces the default SoftwareApplication block). */
  jsonLd?: object;
  /** Sitemap priority 0..1. */
  priority: string;
}

const marker = (label: string) => {
  throw new Error(
    `prerender-seo: expected marker not found (${label}). ` +
      `Did index.html change?`
  );
};

/** Replace once; throw if the search string is absent (guards against drift). */
function replaceOnce(
  html: string,
  search: string | RegExp,
  replacement: string,
  label: string
): string {
  if (typeof search === "string") {
    if (!html.includes(search)) marker(label);
    return html.replace(search, () => replacement);
  }
  if (!search.test(html)) marker(label);
  return html.replace(search, () => replacement);
}

/**
 * Replace the `content=`/`href=` value of a <meta>/<link> identified by one of
 * its attributes (e.g. property="og:title"). Format-agnostic: tolerates
 * attributes split across multiple lines and any attribute order.
 */
function setTagValue(
  html: string,
  tag: "meta" | "link",
  idAttr: string,
  idVal: string,
  valueAttr: "content" | "href",
  value: string,
  label: string
): string {
  const tagRe = new RegExp(`<${tag}\\b[^>]*?\\b${idAttr}="${idVal}"[^>]*?>`);
  const m = html.match(tagRe);
  if (!m) marker(label);
  const oldTag = m![0];
  const valRe = new RegExp(`${valueAttr}="[^"]*"`);
  if (!valRe.test(oldTag)) marker(`${label}:${valueAttr}`);
  const newTag = oldTag.replace(valRe, `${valueAttr}="${value}"`);
  return html.replace(oldTag, () => newTag);
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

/** Minimal, on-brand critical CSS for the static prerendered content. */
const CRITICAL_CSS = `
    <style id="seo-prerender-style">
      #seo-content{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#1e293b;max-width:820px;margin:0 auto;padding:2.5rem 1.5rem 4rem;line-height:1.65}
      #seo-content .seo-brand{font-weight:800;font-size:2rem;letter-spacing:-.02em;background:linear-gradient(135deg,#2563eb 0%,#7c3aed 50%,#db2777 100%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
      #seo-content h1{font-size:2rem;font-weight:700;letter-spacing:-.02em;margin:.2em 0 .4em}
      #seo-content h2{font-size:1.4rem;font-weight:700;margin:1.6em 0 .5em}
      #seo-content h3{font-size:1.1rem;font-weight:600;margin:1.2em 0 .4em}
      #seo-content p{margin:.6em 0}
      #seo-content a{color:#2563eb}
      #seo-content code{background:#f1f5f9;padding:.1em .35em;border-radius:4px;font-size:.9em}
      #seo-content pre{background:#1e293b;color:#e2e8f0;padding:1rem;border-radius:8px;overflow:auto}
      #seo-content pre code{background:none;color:inherit;padding:0}
      #seo-content ul{padding-left:1.4rem}
      #seo-content .seo-links{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem;font-weight:600}
    </style>`;

// --- Static body content -----------------------------------------------------

const homeBody = `
      <div id="seo-content">
        <div class="seo-brand">numbl</div>
        <h1>MATLAB-compatible numerical computing in your browser</h1>
        <p>${HOME_DESC}</p>
        <ul>
          <li>Compatible with MATLAB syntax — run existing <code>.m</code> scripts.</li>
          <li>400+ built-in functions: linear algebra, statistics, signal processing, and more.</li>
          <li>2-D and 3-D plotting.</li>
          <li>Runs entirely in your browser — no installation — or as a Node.js CLI (<code>npx numbl</code>).</li>
          <li>Free and open source (Apache 2.0), from the Flatiron Institute.</li>
        </ul>
        <div class="seo-links">
          <a href="/docs">Documentation</a>
          <a href="/docs/getting-started">Getting Started</a>
          <a href="/gallery">Plot Gallery</a>
          <a href="/embed-repl">REPL</a>
        </div>
      </div>`;

function docsIndexBody(): string {
  const items = docsManifest
    .map(
      d =>
        `          <li><a href="/docs/${d.slug}"><strong>${d.title}</strong></a> — ${d.description}</li>`
    )
    .join("\n");
  return `
      <div id="seo-content">
        <div class="seo-brand">numbl</div>
        <h1>numbl Documentation</h1>
        <p>Guides and reference for numbl, the MATLAB-compatible numerical computing environment.</p>
        <ul>
${items}
        </ul>
      </div>`;
}

function docBody(slug: string): string {
  const meta = docsManifest.find(d => d.slug === slug)!;
  const md = readFileSync(join(DOCS_DIR, meta.file), "utf8");
  const rendered = marked.parse(md, { async: false }) as string;
  return `
      <div id="seo-content">
        <div class="seo-brand"><a href="/" style="text-decoration:none">numbl</a></div>
        ${rendered}
      </div>`;
}

const galleryBody = `
      <div id="seo-content">
        <div class="seo-brand">numbl</div>
        <h1>Plot Gallery</h1>
        <p>Example plots created with numbl — line plots, surfaces, images, and more, each with the source code that produced them. numbl provides MATLAB-compatible 2-D and 3-D plotting in the browser.</p>
        <p><a href="/docs/plotting">Read the plotting documentation</a></p>
      </div>`;

// --- Routes -------------------------------------------------------------------

const routes: Route[] = [
  {
    path: "/",
    title: HOME_TITLE,
    description: HOME_DESC,
    bodyHtml: homeBody,
    priority: "1.0",
  },
  {
    path: "/docs",
    title: "Documentation — numbl",
    description:
      "Documentation for numbl: getting started, language features, 400+ built-in functions, plotting, library usage, and differences from MATLAB.",
    bodyHtml: docsIndexBody(),
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "CollectionPage",
      name: "numbl Documentation",
      url: `${SITE}/docs`,
      isPartOf: { "@type": "WebSite", name: "numbl", url: `${SITE}/` },
    },
    priority: "0.9",
  },
  ...docsManifest.map(
    (d): Route => ({
      path: `/docs/${d.slug}`,
      title: `${d.title} — numbl`,
      description: d.description,
      bodyHtml: docBody(d.slug),
      jsonLd: {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        headline: `${d.title} — numbl`,
        description: d.description,
        url: `${SITE}/docs/${d.slug}`,
        isPartOf: { "@type": "WebSite", name: "numbl", url: `${SITE}/` },
      },
      priority: "0.7",
    })
  ),
  {
    path: "/gallery",
    title: "Plot Gallery — numbl",
    description:
      "A gallery of example plots created with numbl, each with its source code: line plots, surfaces, images, and 3-D visualizations in the browser.",
    bodyHtml: galleryBody,
    priority: "0.6",
  },
];

// --- Rendering ----------------------------------------------------------------

const templatePath = join(DIST, "index.html");
if (!existsSync(templatePath)) {
  throw new Error(
    `prerender-seo: ${templatePath} not found. Run "vite build" first.`
  );
}
const template = readFileSync(templatePath, "utf8");

function renderRoute(route: Route): string {
  const url = route.path === "/" ? `${SITE}/` : `${SITE}${route.path}`;
  const t = escapeAttr(route.title);
  const d = escapeAttr(route.description);
  let html = template;

  const eurl = escapeAttr(url);
  html = replaceOnce(
    html,
    /<title>[^<]*<\/title>/,
    `<title>${route.title}</title>`,
    "title"
  );
  html = setTagValue(
    html,
    "meta",
    "name",
    "description",
    "content",
    d,
    "description"
  );
  html = setTagValue(
    html,
    "link",
    "rel",
    "canonical",
    "href",
    eurl,
    "canonical"
  );
  html = setTagValue(
    html,
    "meta",
    "property",
    "og:url",
    "content",
    eurl,
    "og:url"
  );
  html = setTagValue(
    html,
    "meta",
    "property",
    "og:title",
    "content",
    t,
    "og:title"
  );
  html = setTagValue(
    html,
    "meta",
    "property",
    "og:description",
    "content",
    d,
    "og:description"
  );
  html = setTagValue(
    html,
    "meta",
    "name",
    "twitter:title",
    "content",
    t,
    "twitter:title"
  );
  html = setTagValue(
    html,
    "meta",
    "name",
    "twitter:description",
    "content",
    d,
    "twitter:description"
  );

  // Per-route JSON-LD replaces the default SoftwareApplication block.
  if (route.jsonLd) {
    html = replaceOnce(
      html,
      /<script type="application\/ld\+json">[\s\S]*?<\/script>/,
      `<script type="application/ld+json">\n${JSON.stringify(route.jsonLd, null, 2)}\n    </script>`,
      "json-ld"
    );
  }

  // Inject critical CSS + static content into #root.
  html = replaceOnce(
    html,
    "</head>",
    `${CRITICAL_CSS}\n  </head>`,
    "head-close"
  );
  html = replaceOnce(
    html,
    '<div id="root"></div>',
    `<div id="root">${route.bodyHtml}\n    </div>`,
    "root"
  );

  return html;
}

let count = 0;
for (const route of routes) {
  const html = renderRoute(route);
  const outPath =
    route.path === "/"
      ? join(DIST, "index.html")
      : join(DIST, route.path.replace(/^\//, ""), "index.html");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, html);
  count++;
}

// --- Sitemap ------------------------------------------------------------------

const urls = routes
  .map(r => {
    const loc = r.path === "/" ? `${SITE}/` : `${SITE}${r.path}`;
    return `  <url>\n    <loc>${loc}</loc>\n    <priority>${r.priority}</priority>\n  </url>`;
  })
  .join("\n");
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;
writeFileSync(join(DIST, "sitemap.xml"), sitemap);

console.log(`prerender-seo: wrote ${count} route(s) + sitemap.xml`);
