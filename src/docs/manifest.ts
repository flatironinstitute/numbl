/**
 * Metadata for each documentation page. Single source of truth for slugs,
 * titles, and meta descriptions — used by the docs UI, the runtime page
 * metadata, and the build-time SEO prerender (scripts/prerender-seo.ts,
 * which reads the .md files from disk via `file`).
 */
export interface DocMeta {
  slug: string;
  title: string;
  /** One-sentence summary used for <meta name="description">. */
  description: string;
  /** Markdown filename within src/docs/. */
  file: string;
}

export const docsManifest: DocMeta[] = [
  {
    slug: "getting-started",
    title: "Getting Started",
    description:
      "Run numbl in your browser with no installation, or install the CLI with npm. Covers the REPL, running .m files, CLI options, and the optional native addon.",
    file: "getting-started.md",
  },
  {
    slug: "language",
    title: "Language Features",
    description:
      "MATLAB language features supported by numbl: matrices, operators, data types, control flow, functions, cell arrays, structs, and classes.",
    file: "language.md",
  },
  {
    slug: "builtins",
    title: "Built-in Functions",
    description:
      "Reference for numbl's 400+ MATLAB-compatible built-in functions, organized by category: math, linear algebra, strings, statistics, plotting, and more.",
    file: "builtins.md",
  },
  {
    slug: "plotting",
    title: "Plotting",
    description:
      "2-D and 3-D plotting in numbl with MATLAB-compatible commands: line plots, surfaces, images, subplots, and the CLI plot server.",
    file: "plotting.md",
  },
  {
    slug: "library",
    title: "Library Usage",
    description:
      "Use numbl as a JavaScript library: embed the MATLAB-compatible interpreter in Node.js or browser applications via npm.",
    file: "library.md",
  },
  {
    slug: "deploying",
    title: "Deploying Projects",
    description:
      "Deploy a numbl project as a static website with numbl build-site: bundle .m scripts and markdown into a browser-runnable GitHub Pages site.",
    file: "deploying.md",
  },
  {
    slug: "differences",
    title: "Differences from MATLAB",
    description:
      "Known behavioral differences between numbl and MATLAB, plus current limitations: performance, numeric types, path management, and more.",
    file: "differences.md",
  },
];
