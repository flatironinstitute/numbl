/**
 * Node platform setup shared by the numbl CLI and Node library hosts
 * (exported as `initNodePlatform` from "numbl/node"): the native LAPACK
 * addon, the C-JIT backend used by optimization "2", the koffi bridge for
 * native libraries, and the optional mip core package install.
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { homedir } from "os";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import {
  NATIVE_ADDON_EXPECTED_VERSION,
  registerProvider,
} from "./numbl-core/native/lapack-bridge.js";
import { nativeAddonProvider } from "./numbl-core/native/native-provider.js";
import { registerNodeCompileC } from "./numbl-core/executors/jit/compileC.node.js";
import type { NativeBridge } from "./numbl-core/workspace/types.js";
import { unzipToFiles } from "./vfs/unzipToFiles.js";

// ── Package directory & native addon paths ───────────────────────────────────

/** The numbl package root. This module is bundled into dist-cli/cli.js and
 *  dist-lib/node.js, and run from src/ under tsx: one level down in all
 *  three cases. */
export const packageDir = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Variant-specific addon path. Build-addon copies the freshly built
 *  `numbl_addon.node` here so both fast-math and no-fast-math variants
 *  can coexist; the loader picks the one matching the runtime choice. */
export function variantAddonPath(fastMath: boolean): string {
  const name = fastMath
    ? "numbl_addon.fastmath.node"
    : "numbl_addon.nofastmath.node";
  return join(packageDir, "build", "Release", name);
}

/** Load the native LAPACK addon variant if it is built and its version
 *  matches; otherwise report why and keep the JS fallbacks. Returns whether
 *  the addon was installed. */
function loadNativeAddon(
  fastMath: boolean,
  onMessage: (msg: string) => void
): boolean {
  if (process.env.NUMBL_NO_NATIVE) return false;
  const variantPath = variantAddonPath(fastMath);
  const flag = fastMath ? " --fast-math" : "";
  if (!existsSync(variantPath)) {
    onMessage(
      `Warning: native addon not built${fastMath ? " with fast-math" : " with --no-fast-math"}. ` +
        `Run "npx numbl build-addon${flag}" to build it. Using JS fallbacks.`
    );
    return false;
  }
  try {
    const req = createRequire(import.meta.url);
    const addon = req(variantPath);
    const addonVer =
      typeof addon.addonVersion === "function" ? addon.addonVersion() : 0;
    if (addonVer !== NATIVE_ADDON_EXPECTED_VERSION) {
      onMessage(
        `Warning: native addon version mismatch (got ${addonVer}, expected ${NATIVE_ADDON_EXPECTED_VERSION}). ` +
          `Run "npx numbl build-addon${flag}" to rebuild. Using JS fallbacks.`
      );
      return false;
    }
    registerProvider(nativeAddonProvider(addon));
    return true;
  } catch (e) {
    onMessage(
      `Warning: failed to load native addon at ${variantPath}: ${(e as Error).message}. Using JS fallbacks.`
    );
    return false;
  }
}

/** koffi-backed bridge for native shared libraries and the C-JIT, or
 *  undefined when the optional koffi dependency is not installed. */
function loadKoffiBridge(): NativeBridge | undefined {
  try {
    const req = createRequire(import.meta.url);
    const koffi = req("koffi");
    return { load: (path: string) => koffi.load(path), koffi };
  } catch {
    return undefined;
  }
}

// ── MIP core package auto-install ───────────────────────────────────────────

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-numbl/mip-numbl-any.mhl";
const mipCoreDir = join(
  homedir(),
  ".numbl",
  "mip",
  "packages",
  "gh",
  "mip-org",
  "core",
  "mip"
);
const mipCoreSearchPath = join(mipCoreDir, "mip");
const mipCoreStampPath = join(homedir(), ".numbl", "mip", ".last-refreshed");
const MIP_REFRESH_INTERVAL_MS = 60 * 60 * 1000;

function readMipRefreshStamp(): number {
  try {
    const raw = readFileSync(mipCoreStampPath, "utf8");
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

/** Install the mip core package if missing, or refresh it if the last
 *  refresh is over an hour old. Download failures leave any cached copy. */
async function ensureMipCorePackage(
  onMessage: (msg: string) => void
): Promise<void> {
  const installed = existsSync(join(mipCoreSearchPath, "mip.m"));
  const fresh = Date.now() - readMipRefreshStamp() < MIP_REFRESH_INTERVAL_MS;
  if (installed && fresh) return;
  try {
    if (!installed) onMessage("Installing mip core package...");
    const resp = await fetch(MHL_URL);
    if (!resp.ok) return;
    const buf = new Uint8Array(await resp.arrayBuffer());
    const files = unzipToFiles(buf);
    rmSync(mipCoreDir, { recursive: true, force: true });
    for (const f of files) {
      const dest = join(mipCoreDir, f.path);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, f.content);
    }
    mkdirSync(dirname(mipCoreStampPath), { recursive: true });
    writeFileSync(mipCoreStampPath, String(Date.now()));
    if (!installed) onMessage("mip core package installed.");
  } catch {
    // silently ignore download / install failures; previous cache (if any) stays intact
  }
}

// ── Entry point ─────────────────────────────────────────────────────────────

export interface NodePlatformOptions {
  /** Load the native LAPACK/FFTW addon if it is built (default true). The
   *  NUMBL_NO_NATIVE environment variable also disables it. */
  native?: boolean;
  /** Load the addon variant built with -ffast-math (default false). */
  fastMath?: boolean;
  /** Register the C-JIT backend used by optimization "2" (default true).
   *  It also needs `nativeBridge` (koffi) and a C compiler (`cc`, or
   *  NUMBL_CC); without them "2" behaves like "1". */
  cjit?: boolean;
  /** Install or refresh the mip core package under ~/.numbl/mip and return
   *  its directory in `searchPaths` (default false; downloads from GitHub). */
  mip?: boolean;
  /** Receives warnings and progress messages (default console.error). */
  onMessage?: (msg: string) => void;
}

export interface NodePlatform {
  /** Pass to executeCode (or the ExecSession constructor) to enable the
   *  C-JIT and native shared libraries. Undefined when koffi is missing. */
  nativeBridge: NativeBridge | undefined;
  /** Whether the native LAPACK addon was loaded. */
  nativeAddonLoaded: boolean;
  /** Search paths contributed by the platform (the mip core directory). */
  searchPaths: string[];
}

/**
 * Set up numbl's Node platform services. The addon and the C-JIT backend
 * are process-wide, so calling this again only reapplies the settings.
 */
export async function initNodePlatform(
  opts: NodePlatformOptions = {}
): Promise<NodePlatform> {
  const onMessage = opts.onMessage ?? (msg => console.error(msg));
  const searchPaths: string[] = [];
  if (opts.mip) {
    await ensureMipCorePackage(onMessage);
    if (existsSync(mipCoreSearchPath)) searchPaths.push(mipCoreSearchPath);
  }
  const nativeAddonLoaded =
    opts.native !== false && loadNativeAddon(!!opts.fastMath, onMessage);
  // Installs the koffi C-JIT behind the browser-safe stub in compileC.ts;
  // keeping that import Node-only is what lets the browser bundle stay green.
  if (opts.cjit !== false) registerNodeCompileC();
  return { nativeBridge: loadKoffiBridge(), nativeAddonLoaded, searchPaths };
}
