import { unzipToFiles } from "../vfs/unzipToFiles.js";

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-numbl/mip-numbl-any.mhl";
const MIP_SYSTEM_PREFIX = "mip/packages/gh/mip-org/core/mip/";

function proxiedUrl(url: string): string {
  if (/^https:\/\/github\.com\/.+\/releases\/download\/.+/.test(url)) {
    let proxied = url.replace(
      "https://github.com/",
      "https://mip-cors-proxy.figurl.workers.dev/gh/"
    );
    // Cachebust applies only when going through the proxy: the
    // mip-numbl release tag is mutable so the same URL keeps serving
    // whatever was last published, and otherwise the browser / proxy
    // happily return a stale .mhl for hours.
    proxied += proxied.includes("?") ? "&" : "?";
    proxied += "t=" + Date.now();
    return proxied;
  }
  return url;
}

export interface VfsFile {
  path: string;
  content: Uint8Array;
}

/**
 * Fetches the mip core package from GitHub, unzips it, and returns
 * VFS-ready files with /system/ prefix paths. The URL goes through
 * `proxiedUrl`, which appends a per-call `?t=<ms>` cachebust when (and
 * only when) it rewrites a GitHub-release URL onto the CORS proxy.
 */
export async function fetchMipCoreFiles(): Promise<VfsFile[]> {
  const resp = await fetch(proxiedUrl(MHL_URL));
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const extracted = unzipToFiles(new Uint8Array(buf));
  return extracted.map(f => ({
    path: "/system/" + MIP_SYSTEM_PREFIX + f.path,
    content: f.content,
  }));
}
