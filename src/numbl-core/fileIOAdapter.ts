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

export interface FileIOAdapter {
  /** Open a file, returns an integer file identifier (fid). */
  fopen(filename: string, permission: string): number;

  /** Close a file (or 'all'). Returns 0 on success, -1 on error. */
  fclose(fid: number | "all"): number;

  /** Read one line, stripping the newline. Returns -1 at EOF. */
  fgetl(fid: number): string | number;

  /** Read one line, keeping the newline. Returns -1 at EOF. */
  fgets(fid: number): string | number;

  /** Read an entire file as a string. */
  fileread(filename: string): string;

  /** Returns 1 if at EOF, 0 otherwise. */
  feof(fid: number): number;

  /** Returns the error message string for the last operation on fid. */
  ferror(fid: number): string;

  /** Write a string to a file descriptor (used by fprintf). */
  fwrite(fid: number, text: string): void;

  /** Read raw bytes from a file. Returns the bytes actually read (may be fewer than count at EOF). Optional. */
  freadBytes?(fid: number, count: number): Uint8Array;

  /** Write raw bytes to a file. Returns number of bytes written. Optional. */
  fwriteBytes?(fid: number, data: Uint8Array): number;

  /** Seek to a position in a file. origin: -1=bof, 0=cof, 1=eof. Returns 0 on success, -1 on error. Optional. */
  fseek?(fid: number, offset: number, origin: number): number;

  /** Return current file position. Optional. */
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

  /** Extract a ZIP file to an output folder. Returns list of extracted file paths. Optional. */
  unzip?(zipfilename: string, outputfolder: string): string[];

  /** Return the temporary directory path. Optional. */
  tempdir?(): string;

  /** List directory entries. Returns array of {name, folder, bytes, isdir, mtimeMs}. Optional. */
  listDir?(dirPath: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[];
}
