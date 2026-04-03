import { inflateSync } from "fflate";

const TEXT_DECODER = new TextDecoder("utf-8");

/**
 * Unzip a ZIP archive from a Uint8Array and return extracted files.
 */
export function unzipToFiles(
  zipData: Uint8Array
): { path: string; content: Uint8Array }[] {
  // Find End of Central Directory record (signature 0x06054b50)
  let eocdOffset = -1;
  for (let i = zipData.length - 22; i >= 0; i--) {
    if (
      zipData[i] === 0x50 &&
      zipData[i + 1] === 0x4b &&
      zipData[i + 2] === 0x05 &&
      zipData[i + 3] === 0x06
    ) {
      eocdOffset = i;
      break;
    }
  }
  if (eocdOffset === -1) throw new Error("Invalid ZIP file");

  const view = new DataView(
    zipData.buffer,
    zipData.byteOffset,
    zipData.byteLength
  );
  const cdOffset = view.getUint32(eocdOffset + 16, true);
  const cdEntries = view.getUint16(eocdOffset + 10, true);
  const files: { path: string; content: Uint8Array }[] = [];
  let pos = cdOffset;

  for (let i = 0; i < cdEntries; i++) {
    if (view.getUint32(pos, true) !== 0x02014b50) {
      throw new Error("Corrupt ZIP central directory");
    }
    const method = view.getUint16(pos + 10, true);
    const cdCompressedSize = view.getUint32(pos + 20, true);
    const nameLen = view.getUint16(pos + 28, true);
    const extraLen = view.getUint16(pos + 30, true);
    const commentLen = view.getUint16(pos + 32, true);
    const localHeaderOffset = view.getUint32(pos + 42, true);
    const entryName = TEXT_DECODER.decode(
      zipData.subarray(pos + 46, pos + 46 + nameLen)
    );
    pos += 46 + nameLen + extraLen + commentLen;

    if (entryName.endsWith("/")) continue; // directory entry

    const localPos = localHeaderOffset;
    if (view.getUint32(localPos, true) !== 0x04034b50) {
      throw new Error("Corrupt ZIP local header");
    }
    const localNameLen = view.getUint16(localPos + 26, true);
    const localExtraLen = view.getUint16(localPos + 28, true);
    const localCompressedSize = view.getUint32(localPos + 18, true);
    // Use central directory size when local header has 0 (data descriptor flag)
    const compressedSize = localCompressedSize || cdCompressedSize;
    const dataStart = localPos + 30 + localNameLen + localExtraLen;
    const compressedData = zipData.subarray(
      dataStart,
      dataStart + compressedSize
    );

    let content: Uint8Array;
    if (method === 0) {
      content = new Uint8Array(compressedData);
    } else if (method === 8) {
      content = inflateSync(compressedData);
    } else {
      throw new Error(
        `Unsupported ZIP compression method ${method} for ${entryName}`
      );
    }

    files.push({ path: entryName, content });
  }

  return files;
}
