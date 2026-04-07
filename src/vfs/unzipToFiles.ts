import { unzipSync } from "fflate";

/**
 * Unzip a ZIP archive from a Uint8Array and return extracted files.
 */
export function unzipToFiles(
  zipData: Uint8Array
): { path: string; content: Uint8Array }[] {
  const files = unzipSync(zipData);
  const out: { path: string; content: Uint8Array }[] = [];
  for (const [path, content] of Object.entries(files)) {
    if (path.endsWith("/")) continue; // skip directory entries
    out.push({ path, content });
  }
  return out;
}
