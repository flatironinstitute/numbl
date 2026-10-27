import { useState, useEffect } from "react";
import type { UseProjectFilesResult } from "./useProjectFiles";
import type { FileEntry } from "./projectFilesState";
import { useInMemoryProjectFiles } from "./useInMemoryProjectFiles";
import { unzipToFiles } from "../vfs/unzipToFiles";

/**
 * One end-user "figure view": a named, editor-less route (`#figure/<id>`) that
 * auto-runs `entry` and shows its output then its (interactive) figure. A
 * project can declare several; the IDE route is unaffected.
 */
export interface FigureSpec {
  /** Stable, URL-safe slug used in the `#figure/<id>` route. */
  id: string;
  /** Human label shown in the figure view header / browser tab. */
  name?: string;
  /** The `.m` script the figure view runs (relative to the bundle root). */
  entry: string;
}

/**
 * Optional manifest baked into the project bundle. When present (as
 * `numbl-project.json` at the bundle root) it controls the entry file and
 * site title. The manifest itself is hidden from the file browser.
 */
interface ProjectManifest {
  entry?: string;
  title?: string;
  repository?: string;
  /** Cap (px) on the initial output-panel height; gives the figure panel
   *  below it more room on first load (desktop layout). */
  maxInitialOutputPanelHeight?: number;
  /** End-user figure views (see FigureSpec). */
  figures?: FigureSpec[];
}

const MANIFEST_NAME = "numbl-project.json";

/** Build id injected by `build-site` (a hash of the bundle). Used to
 *  cache-bust the project.zip fetch per build, without busting on every
 *  reload. Undefined in dev / older bundles. */
function getBuildId(): string | undefined {
  return typeof window !== "undefined"
    ? (window as unknown as { __NUMBL_BUILD_ID__?: string }).__NUMBL_BUILD_ID__
    : undefined;
}

/** Resolve the deploy base path (where `project.zip` lives). */
function getSiteBase(): string {
  // The build-site CLI injects an absolute base for GitHub Pages project
  // sites (served under /<repo>/). Falls back to Vite's BASE_URL ("./" for
  // the relative-base build, "/" in dev), which resolves against the
  // single-page document root.
  const injected =
    typeof window !== "undefined"
      ? (window as unknown as { __NUMBL_BASE__?: string }).__NUMBL_BASE__
      : undefined;
  const base = injected || import.meta.env.BASE_URL || "/";
  return base.endsWith("/") ? base : base + "/";
}

/** Where to look for `project.zip`, in order.
 *
 *  The injected base is tried first: it is absolute, so it holds even on a
 *  host that serves index.html for a path without a trailing slash, where a
 *  relative URL would resolve one directory too high. It is wrong, however,
 *  when the site is served somewhere the build did not know about: under a
 *  custom domain at the root, in a directory moved by hand, or from a git
 *  vault that serves pages at /<collection>/<repo>/pages/ rather than at
 *  /<repo>/. The document-relative URL is right in those cases, so we try it
 *  as a fallback instead of reporting the bundle as missing. */
function projectZipUrls(): string[] {
  const buildId = getBuildId();
  const file =
    "project.zip" + (buildId ? `?v=${encodeURIComponent(buildId)}` : "");
  const primary = getSiteBase() + file;
  const relative = "./" + file;
  const absolute = (u: string) =>
    typeof document !== "undefined" ? new URL(u, document.baseURI).href : u;
  return absolute(primary) === absolute(relative)
    ? [primary]
    : [primary, relative];
}

/** Fetch the deployed bundle, trying each candidate location in turn. */
async function fetchProjectZip(): Promise<Uint8Array> {
  const urls = projectZipUrls();
  let lastError: unknown;
  for (let i = 0; i < urls.length; i++) {
    try {
      const resp = await fetch(urls[i]);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return new Uint8Array(await resp.arrayBuffer());
    } catch (e) {
      lastError = e;
      if (i + 1 < urls.length) {
        console.warn(
          `project bundle not at ${urls[i]} (${e instanceof Error ? e.message : e}); trying ${urls[i + 1]}`
        );
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Choose a sensible landing file: manifest entry, else README, else a script. */
function pickActiveFile(files: FileEntry[], entry?: string): string {
  if (entry) {
    const m = files.find(f => f.name === entry);
    if (m) return m.id;
  }
  const readme = files.find(f => /(^|\/)README\.md$/i.test(f.name));
  if (readme) return readme.id;
  const main = files.find(f => /(^|\/)main\.m$/i.test(f.name));
  if (main) return main.id;
  const firstScript = files.find(f => f.name.endsWith(".m"));
  if (firstScript) return firstScript.id;
  return files.length > 0 ? files[0].id : "";
}

export interface UseStaticProjectFilesResult extends UseProjectFilesResult {
  /** Title from the bundle manifest, if any. */
  title: string | null;
  /** Source repository URL from the bundle manifest, if any. */
  repository: string | null;
  /** Cap (px) on the initial output-panel height from the manifest, if any. */
  maxInitialOutputPanelHeight: number | null;
  /** End-user figure views declared in the manifest (empty if none). */
  figures: FigureSpec[];
  /** True if the bundle failed to load. */
  loadError: string | null;
}

/**
 * File source for a statically-deployed numbl project. On mount it fetches
 * `project.zip` from the deploy base, unzips it (binary-safe) into an
 * in-memory content map, and exposes the standard project-files interface.
 *
 * Edits live in memory for the session and reset on reload — the deployed
 * bundle is the source of truth. Like useShareProjectFiles, but loads from a
 * baked zip rather than the URL hash so binary files survive intact.
 */
export function useStaticProjectFiles(): UseStaticProjectFilesResult {
  const project = useInMemoryProjectFiles();
  const { setProject, setLoading } = project;
  const [title, setTitle] = useState<string | null>(null);
  const [repository, setRepository] = useState<string | null>(null);
  const [maxInitialOutputPanelHeight, setMaxInitialOutputPanelHeight] =
    useState<number | null>(null);
  const [figures, setFigures] = useState<FigureSpec[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const buf = await fetchProjectZip();
        if (cancelled) return;

        const extracted = unzipToFiles(buf);

        // Pull out the optional manifest before building the file list.
        let manifest: ProjectManifest = {};
        const manifestEntry = extracted.find(f => f.path === MANIFEST_NAME);
        if (manifestEntry) {
          try {
            manifest = JSON.parse(
              new TextDecoder().decode(manifestEntry.content)
            );
          } catch {
            // Malformed manifest is non-fatal; ignore it.
          }
        }

        const wsFiles: FileEntry[] = [];
        const contentMap = new Map<string, Uint8Array>();
        for (const f of extracted) {
          if (f.path === MANIFEST_NAME) continue;
          const id = crypto.randomUUID();
          wsFiles.push({ id, name: f.path });
          contentMap.set(id, f.content);
        }
        wsFiles.sort((a, b) => a.name.localeCompare(b.name));

        setProject(
          wsFiles,
          contentMap,
          pickActiveFile(wsFiles, manifest.entry)
        );
        setTitle(manifest.title ?? null);
        setRepository(manifest.repository ?? null);
        setMaxInitialOutputPanelHeight(
          typeof manifest.maxInitialOutputPanelHeight === "number"
            ? manifest.maxInitialOutputPanelHeight
            : null
        );
        setFigures(
          Array.isArray(manifest.figures)
            ? manifest.figures.filter(
                f =>
                  f && typeof f.id === "string" && typeof f.entry === "string"
              )
            : []
        );
      } catch (e) {
        if (cancelled) return;
        console.error("Failed to load project bundle:", e);
        setLoadError(e instanceof Error ? e.message : "Failed to load project");
        const id = crypto.randomUUID();
        setProject(
          [{ id, name: "script.m" }],
          new Map([
            [id, new TextEncoder().encode("% Failed to load project bundle\n")],
          ]),
          id
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setProject, setLoading]);

  return {
    ...project,
    title,
    repository,
    maxInitialOutputPanelHeight,
    figures,
    loadError,
  };
}
