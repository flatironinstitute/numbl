/**
 * mip bootstrap for embedded sessions: fetch the mip core package from its
 * GitHub release and lay it out at the same system prefix the numbl IDE
 * uses, so `mip load --install <pkg>` works inside the session script.
 * (Same semantics as src/hooks/fetchMipCoreFiles.ts, which serves the IDE's
 * IndexedDB-backed system directory; this one is host-framework-free.)
 */
import { unzipToFiles } from "../vfs/unzipToFiles.js";

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-numbl/mip-numbl-any.mhl";

export const MIP_SYSTEM_PREFIX = "/system/mip/packages/gh/mip-org/core/mip/";

/** Directory that must be on the search path for `mip` to resolve. */
export const MIP_SEARCH_PATH = MIP_SYSTEM_PREFIX + "mip";

/** Present iff mip core is installed in the VFS. */
export const MIP_MARKER_PATH = MIP_SYSTEM_PREFIX + "mip/mip.m";

function proxiedUrl(url: string): string {
  // GitHub release assets lack CORS headers — route through the proxy. The
  // cachebust matters: the mip-numbl release tag is mutable, and without it
  // the browser/proxy happily serve a stale .mhl for hours.
  if (/^https:\/\/github\.com\/.+\/releases\/download\/.+/.test(url)) {
    let proxied = url.replace(
      "https://github.com/",
      "https://mip-cors-proxy.figurl.workers.dev/gh/"
    );
    proxied += proxied.includes("?") ? "&" : "?";
    proxied += "t=" + Date.now();
    return proxied;
  }
  return url;
}

export interface MipCoreFile {
  path: string;
  content: Uint8Array;
}

export async function fetchMipCoreFiles(): Promise<MipCoreFile[]> {
  const resp = await fetch(proxiedUrl(MHL_URL));
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching mip core`);
  const extracted = unzipToFiles(new Uint8Array(await resp.arrayBuffer()));
  return extracted.map(f => ({
    path: MIP_SYSTEM_PREFIX + f.path,
    content: f.content,
  }));
}
