import { unzipToFiles } from "../vfs/unzipToFiles.js";

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-main/mip-main-any.mhl";
const MIP_SYSTEM_PREFIX = ".mip/packages/mip-org/core/mip/";

function proxiedUrl(url: string): string {
  if (/^https:\/\/github\.com\/.+\/releases\/download\/.+/.test(url)) {
    return url.replace(
      "https://github.com/",
      "https://mip-cors-proxy.figurl.workers.dev/gh/"
    );
  }
  return url;
}

export interface VfsFile {
  path: string;
  content: Uint8Array;
}

/**
 * Fetches the mip core package from GitHub, unzips it, and returns
 * VFS-ready files with /system/ prefix paths.
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
