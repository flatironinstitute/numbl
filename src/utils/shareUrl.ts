import pako from "pako";
import { fileText, type WorkspaceFile } from "../hooks/useProjectFiles";

export interface ShareData {
  files: { name: string; content: string }[];
  activeFileName: string | null;
}

const textEncoder = new TextEncoder();

export function encodeShareData(
  files: WorkspaceFile[],
  contentMap: Map<string, Uint8Array>,
  activeFileId: string
): string {
  const activeFile = files.find(f => f.id === activeFileId);
  const data: ShareData = {
    files: files.map(f => ({
      name: f.name,
      content: fileText(contentMap.get(f.id) ?? new Uint8Array(0)),
    })),
    activeFileName: activeFile?.name ?? null,
  };

  const json = JSON.stringify(data);
  const compressed = pako.deflate(textEncoder.encode(json));

  // Base64url encoding (no padding, URL-safe)
  let base64 = btoa(String.fromCharCode(...compressed));
  base64 = base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

  return base64;
}

/** Encode a single script into a share-page URL hash. */
export function makeShareHash(name: string, code: string): string {
  const data: ShareData = {
    files: [{ name: `${name}.m`, content: code }],
    activeFileName: `${name}.m`,
  };
  const json = JSON.stringify(data);
  const compressed = pako.deflate(textEncoder.encode(json));
  let base64 = btoa(String.fromCharCode(...compressed));
  base64 = base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return base64;
}

export function decodeShareData(encoded: string): ShareData {
  // Base64url decoding
  let base64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
  // Add padding
  while (base64.length % 4) base64 += "=";

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  const decompressed = pako.inflate(bytes);
  const json = new TextDecoder().decode(decompressed);
  return JSON.parse(json);
}

export interface ShareFilesResult {
  files: WorkspaceFile[];
  contentMap: Map<string, Uint8Array>;
  activeFileId: string;
}

export function shareDataToWorkspaceFiles(data: ShareData): ShareFilesResult {
  const files: WorkspaceFile[] = [];
  const contentMap = new Map<string, Uint8Array>();

  for (const f of data.files) {
    const id = crypto.randomUUID();
    files.push({ id, name: f.name });
    contentMap.set(id, textEncoder.encode(f.content));
  }

  let activeFileId = "";
  if (data.activeFileName) {
    const match = files.find(f => f.name === data.activeFileName);
    if (match) activeFileId = match.id;
  }
  if (!activeFileId && files.length > 0) {
    activeFileId = files[0].id;
  }

  return { files, contentMap, activeFileId };
}
