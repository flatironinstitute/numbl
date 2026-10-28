/**
 * Platform-agnostic file I/O adapter interface.
 *
 * numbl-core defines this interface but never imports Node.js fs.
 * Concrete implementations (e.g. NodeFileIOAdapter) are injected
 * from the CLI or other host environments via ExecOptions.fileIO.
 */

/** Options extracted from a weboptions struct for websave/webread. */
export interface WebOptions {
  timeout?: number;
  requestMethod?: string;
  headerFields?: [string, string][];
  username?: string;
  password?: string;
  keyName?: string;
  keyValue?: string;
}

/** Subset of MATLAB's fileattrib attribute struct. `Name` is the resolved
 *  absolute path. */
export interface FileAttribInfo {
  Name: string;
  directory: boolean;
  UserRead: boolean;
  UserWrite: boolean;
  UserExecute: boolean;
}

/**
 * Random access to open files. Hosts that provide it (Node) let open files
 * be read and written in place with bounded memory; without it the core
 * keeps each open file in memory.
 */
export interface FileHandleIO {
  /** Open an existing file for reading ('r') or reading and writing
   *  ('r+'). Throws (with a Node-style `code`) on failure. */
  openHandle(path: string, mode: "r" | "r+"): number;
  /** Up to `len` bytes at `pos`; fewer only at end of file. */
  readAt(handle: number, pos: number, len: number): Uint8Array;
  writeAt(handle: number, pos: number, bytes: Uint8Array): void;
  size(handle: number): number;
  /** Make written data visible to other readers of the file, if the host
   *  buffers it. Optional. */
  flush?(handle: number): void;
  close(handle: number): void;
}

/**
 * Host filesystem access. numbl-core owns all stream semantics (fids,
 * modes, positions, EOF, encodings, buffering; see `FileTable`) and only
 * needs byte primitives from the host: `readFile`, `writeFile`,
 * `existsPath`, `resolvePath`, and optionally `handles`. The remaining
 * methods back individual builtins (dir, mkdir, copyfile, zip, websave, ...).
 */
export interface FileIOAdapter {
  /** Random access to open files. Optional. */
  handles?: FileHandleIO;

  /** Read a whole file. Throws if it cannot be read. */
  readFile?(path: string): Uint8Array;

  /** Create or replace a file with `data`. Throws on failure; an error
   *  with a Node-style `code` ('ENOENT', 'EACCES', ...) gives fopen its
   *  message. */
  writeFile?(path: string, data: Uint8Array): void;

  /** Set the user write/execute permission of a file or folder (the
   *  `fileattrib(name, '+w')` form). Returns true on success. Optional. */
  setFileMode?(
    path: string,
    mode: { write?: boolean; execute?: boolean }
  ): boolean;

  // ── Legacy stream API ───────────────────────────────────────────────
  // Hosts written before the core FileTable implemented these instead of
  // readFile/writeFile. numbl-core no longer calls them except to emulate
  // readFile/writeFile for such hosts.

  /** @deprecated implement readFile/writeFile instead. */
  fopen?(filename: string, permission: string): number;
  /** @deprecated */
  fclose?(fid: number | "all"): number;
  /** @deprecated */
  fgetl?(fid: number): string | number;
  /** @deprecated */
  fgets?(fid: number): string | number;
  /** @deprecated implement readFile instead. */
  fileread?(filename: string): string;
  /** @deprecated */
  feof?(fid: number): number;
  /** @deprecated */
  ferror?(fid: number): string;
  /** @deprecated */
  fwrite?(fid: number, text: string): void;
  /** @deprecated */
  freadBytes?(fid: number, count: number): Uint8Array;
  /** @deprecated */
  fwriteBytes?(fid: number, data: Uint8Array): number;
  /** @deprecated */
  fseek?(fid: number, offset: number, origin: number): number;
  /** @deprecated */
  ftell?(fid: number): number;

  /** Scan a directory for workspace files (.m, .js, .wasm). Optional. */
  scanDirectory?(
    dirPath: string
  ): import("../numbl-core/workspace/index.js").WorkspaceFile[];

  /** Resolve a path to absolute. Optional. */
  resolvePath?(dirPath: string): string;

  /** Check whether a path exists and whether it is a file or directory. Optional. */
  existsPath?(path: string): "file" | "dir" | null;

  /** Create a directory (and parents). Returns true on success. Optional. */
  mkdir?(dirPath: string): boolean;

  /** Download a URL to a file. Optional. */
  websave?(url: string, filename: string, options?: WebOptions): void;

  /** Fetch content from a URL and return as a string. Optional. */
  webread?(url: string, options?: WebOptions): string;

  /** Delete files matching a pattern (supports globs). Optional. */
  deleteFile?(pattern: string): void;

  /** Remove a directory. If recursive is true, remove contents first. Returns true on success. Optional. */
  rmdir?(dirPath: string, recursive: boolean): boolean;

  /** Move or rename a file or folder. If force is true, overwrites read-only files. Returns true on success. Optional. */
  movefile?(source: string, destination: string, force: boolean): boolean;

  /** Copy a file or folder. If force is true, overwrites read-only files. Returns true on success. Optional. */
  copyfile?(source: string, destination: string, force: boolean): boolean;

  /** Resolve a file/dir to its absolute path and attributes, or null if it does not exist. Optional. */
  fileattrib?(path: string): FileAttribInfo | null;

  /** Extract a ZIP file to an output folder. Returns list of extracted file paths. Optional. */
  unzip?(zipfilename: string, outputfolder: string): string[];

  /** Create a ZIP file from files/folders (paths resolved relative to
   *  rootDir; folders are added recursively; simple globs in the last
   *  path component are expanded). Returns the entry names written. Optional. */
  zip?(zipfilename: string, filenames: string[], rootDir: string): string[];

  /** Return the temporary directory path. Optional. */
  tempdir?(): string;

  /** Return the numbl user/config directory path. Optional. */
  userpath?(): string;

  /** List directory entries. Returns array of {name, folder, bytes, isdir, mtimeMs}. Optional. */
  listDir?(dirPath: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[];
}
