/**
 * Filesystem builtins on the runtime's FileIOAdapter: mkdir, delete, rmdir,
 * movefile, copyfile, fileattrib, zip, unzip, dir, tempdir, userpath,
 * tempname.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  RuntimeError,
} from "../../runtime/index.js";
import { isRuntimeCell, isRuntimeGraphicsHandle } from "../../runtime/types.js";
import type { Runtime } from "../../runtime/runtime.js";
import type { FileIOAdapter } from "../../fileIOAdapter.js";
import { builtin } from "./builtin.js";

/** The runtime's file I/O adapter, or an error when file I/O is not
 *  available. */
export function requireFileIO(rt: Runtime): FileIOAdapter {
  const io = rt.fileIO;
  if (!io)
    throw new RuntimeError("File I/O is not available in this environment");
  return io;
}

builtin({
  name: "mkdir",
  nargin: [1, 2],
  nargout: 3,
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.mkdir)
      throw new RuntimeError("mkdir is not available in this environment");
    // mkdir(folder) or mkdir(parentFolder, folder)
    const dirPath =
      args.length === 1
        ? toString(args[0])
        : toString(args[0]) + "/" + toString(args[1]);
    const ok = io.mkdir(dirPath);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`mkdir: cannot create directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.char(ok ? "true" : "false")
      : [
          RTV.char(ok ? "true" : "false"),
          RTV.char(ok ? "" : `Cannot create directory '${dirPath}'`),
          RTV.char(""),
        ];
  },
});

builtin({
  name: "delete",
  nargin: [1, Infinity],
  nargout: 0,
  apply: (args, _nargout, ctx) => {
    const rt = ctx.rt;
    for (const arg of args) {
      // delete(handle): remove the graphics object's plot instruction.
      if (isRuntimeGraphicsHandle(arg)) {
        const instr = arg._trace.__instruction;
        if (instr) {
          const idx = rt.plotInstructions.indexOf(instr as never);
          if (idx >= 0) rt.plotInstructions.splice(idx, 1);
        }
        continue;
      }
      // delete(filename): remove a file.
      const io = requireFileIO(rt);
      if (!io.deleteFile)
        throw new RuntimeError("delete is not available in this environment");
      io.deleteFile(toString(arg));
    }
  },
});

builtin({
  name: "rmdir",
  nargin: [1, 2],
  nargout: 3,
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.rmdir)
      throw new RuntimeError("rmdir is not available in this environment");
    const dirPath = toString(args[0]);
    const recursive =
      args.length >= 2 && toString(args[1]).toLowerCase() === "s";
    const ok = io.rmdir(dirPath, recursive);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`rmdir: cannot remove directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.num(ok ? 1 : 0)
      : [
          RTV.num(ok ? 1 : 0),
          RTV.char(ok ? "" : `Cannot remove directory '${dirPath}'`),
          RTV.char(""),
        ];
  },
});

// movefile / copyfile(source[, destination[, 'f']]). With one argument the
// destination is the current folder. A third positional 'f' forces the
// overwrite; MATLAB's name=value options are not supported.
for (const [name, verb] of [
  ["movefile", "move"],
  ["copyfile", "copy"],
] as const) {
  builtin({
    name,
    nargin: [1, Infinity],
    nargout: 3,
    apply: (args, nargout, ctx) => {
      const rt = ctx.rt;
      const io = requireFileIO(rt);
      const op = io[name];
      if (!op)
        throw new RuntimeError(`${name} is not available in this environment`);
      const source = toString(args[0]);
      const destination =
        args.length >= 2 ? toString(args[1]) : (rt.system?.cwd() ?? ".");
      const force = args.length >= 3 && toString(args[2]).toLowerCase() === "f";
      const ok = op.call(io, source, destination, force);
      if (nargout === 0) {
        if (!ok)
          throw new RuntimeError(
            `${name}: cannot ${verb} '${source}' to '${destination}'`
          );
        return undefined;
      }
      return nargout <= 1
        ? RTV.num(ok ? 1 : 0)
        : [
            RTV.num(ok ? 1 : 0),
            RTV.char(
              ok ? "" : `Cannot ${verb} '${source}' to '${destination}'`
            ),
            RTV.char(""),
          ];
    },
  });
}

/** fileattrib(name, attribs[, users]): only the user write/execute bits
 *  ('+w', '-w', '+x', '-x', space-separated) are supported. */
function setFileAttrib(
  io: FileIOAdapter,
  path: string,
  attribs: string,
  nargout: number
): RuntimeValue | RuntimeValue[] | undefined {
  const mode: { write?: boolean; execute?: boolean } = {};
  for (const tok of attribs.split(/\s+/).filter(t => t)) {
    const m = /^([+-])([wx])$/.exec(tok);
    if (!m)
      throw new RuntimeError(`fileattrib: unsupported attribute '${tok}'`);
    mode[m[2] === "w" ? "write" : "execute"] = m[1] === "+";
  }
  if (!io.setFileMode)
    throw new RuntimeError("fileattrib is not available in this environment");
  const ok = io.setFileMode(path, mode);
  const msg = ok ? "" : `No such file or directory: ${path}`;
  if (nargout === 0) {
    if (!ok) throw new RuntimeError(`fileattrib: ${msg}`);
    return undefined;
  }
  return nargout <= 1
    ? RTV.num(ok ? 1 : 0)
    : [RTV.num(ok ? 1 : 0), RTV.char(msg), RTV.char("")];
}

builtin({
  name: "fileattrib",
  // MATLAB also accepts no arguments (the current folder), which numbl
  // does not support.
  nargin: [0, 4],
  nargout: 3,
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.fileattrib)
      throw new RuntimeError("fileattrib is not available in this environment");
    if (args.length < 1)
      throw new RuntimeError("fileattrib requires at least 1 argument");
    const p = toString(args[0]);
    if (args.length >= 2)
      return setFileAttrib(io, p, toString(args[1]), nargout);
    const info = io.fileattrib(p);
    if (nargout === 0) {
      if (!info)
        throw new RuntimeError(
          `fileattrib: cannot find '${p}': No such file or directory`
        );
      return undefined;
    }
    const values: RuntimeValue = info
      ? RTV.struct(
          new Map<string, RuntimeValue>([
            ["Name", RTV.char(info.Name)],
            ["archive", RTV.num(0)],
            ["system", RTV.num(0)],
            ["hidden", RTV.num(0)],
            ["directory", RTV.num(info.directory ? 1 : 0)],
            ["UserRead", RTV.num(info.UserRead ? 1 : 0)],
            ["UserWrite", RTV.num(info.UserWrite ? 1 : 0)],
            ["UserExecute", RTV.num(info.UserExecute ? 1 : 0)],
          ])
        )
      : RTV.char(`No such file or directory: ${p}`);
    if (nargout <= 1) return RTV.num(info ? 1 : 0);
    return [RTV.num(info ? 1 : 0), values, RTV.char("")];
  },
});

builtin({
  name: "unzip",
  nargin: [1, Infinity],
  mfile: true,
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.unzip)
      throw new RuntimeError("unzip is not available in this environment");

    let zipfilename = toString(args[0]);
    let outputfolder = ".";
    let nextArg = 1;

    // Second positional arg is outputfolder (if not a name-value pair)
    if (args.length >= 2 && toString(args[1]) !== "Password") {
      outputfolder = toString(args[1]);
      nextArg = 2;
    }

    // Check for Password name-value pair (not supported, but give clear error)
    for (let i = nextArg; i < args.length; i += 2) {
      const name = toString(args[i]);
      if (name === "Password") {
        throw new RuntimeError(
          "unzip: Password-protected ZIP files are not supported"
        );
      }
    }

    // If no extension, try appending .zip
    if (!zipfilename.includes(".")) {
      zipfilename = zipfilename + ".zip";
    }

    // Support URL input: download to a temp file first
    let tempFile: string | null = null;
    if (
      zipfilename.startsWith("http://") ||
      zipfilename.startsWith("https://")
    ) {
      if (!io.websave)
        throw new RuntimeError(
          "unzip: URL download is not available in this environment"
        );
      // Create a temp file path in the output folder
      if (io.mkdir) io.mkdir(outputfolder);
      tempFile = outputfolder + "/.numbl_unzip_tmp_" + Date.now() + ".zip";
      io.websave(zipfilename, tempFile);
      zipfilename = tempFile;
    }

    let extracted: string[];
    try {
      extracted = io.unzip(zipfilename, outputfolder);
    } finally {
      // Clean up temp file
      if (tempFile && io.deleteFile) {
        try {
          io.deleteFile(tempFile);
        } catch {
          // ignore cleanup errors
        }
      }
    }

    if (nargout >= 1) {
      // Return cell array of extracted file names
      const cellData: RuntimeValue[] = extracted.map(f => RTV.char(f));
      return RTV.cell(cellData, [1, cellData.length]);
    }
    return undefined;
  },
});

builtin({
  name: "zip",
  nargin: [2, Infinity],
  mfile: true,
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.zip)
      throw new RuntimeError("zip is not available in this environment");

    // MATLAB appends .zip when the target filename has no extension
    let zipfilename = toString(args[0]);
    const base = zipfilename.split(/[/\\]/).pop() ?? zipfilename;
    if (!base.includes(".")) zipfilename += ".zip";

    const fnames = args[1];
    const filenames: string[] = isRuntimeCell(fnames)
      ? fnames.data.map(v => toString(v))
      : [toString(fnames)];
    const rootdir = args.length >= 3 ? toString(args[2]) : ".";

    const entryNames = io.zip(zipfilename, filenames, rootdir);

    if (nargout >= 1) {
      const cellData: RuntimeValue[] = entryNames.map(f => RTV.char(f));
      return RTV.cell(cellData, [1, cellData.length]);
    }
    return undefined;
  },
});

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

builtin({
  name: "dir",
  nargin: [0, 1],
  apply: (args, nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.listDir)
      throw new RuntimeError("dir is not available in this environment");
    const pattern = args.length >= 1 ? toString(args[0]) : ".";

    const entries = io.listDir(pattern);

    if (nargout === 0) {
      // Display mode: print names separated by spaces
      const names = entries.map(e => e.name);
      if (names.length > 0) {
        ctx.rt.output(names.join("  ") + "\n");
      }
      return undefined;
    }

    // Return struct array with fields: name, folder, date, bytes, isdir, datenum
    const fieldNames = ["name", "folder", "date", "bytes", "isdir", "datenum"];
    const elements = entries.map(e => {
      const d = new Date(e.mtimeMs);
      // Format date like MATLAB: dd-Mon-yyyy HH:MM:SS
      const dd = String(d.getDate()).padStart(2, "0");
      const mon = MONTHS[d.getMonth()];
      const yyyy = d.getFullYear();
      const HH = String(d.getHours()).padStart(2, "0");
      const MM = String(d.getMinutes()).padStart(2, "0");
      const SS = String(d.getSeconds()).padStart(2, "0");
      const dateStr = `${dd}-${mon}-${yyyy} ${HH}:${MM}:${SS}`;
      // datenum: MATLAB serial date number (days since 0-Jan-0000)
      // MATLAB epoch offset: 719529 days from 0-Jan-0000 to 1-Jan-1970.
      // MATLAB's dir returns datenum as a *local-calendar* serial number
      // (so converting it back via datetime(...,'ConvertFrom','datenum')
      // yields the same clock time as the 'date' field). Shift the UTC
      // epoch-ms by the local TZ offset so that interpreting the result
      // as UTC calendar recovers the host's local calendar value.
      const datenum =
        719529 + (e.mtimeMs - d.getTimezoneOffset() * 60000) / 86400000;

      return RTV.struct(
        new Map<string, RuntimeValue>([
          ["name", RTV.char(e.name)],
          ["folder", RTV.char(e.folder)],
          ["date", RTV.char(dateStr)],
          ["bytes", RTV.num(e.bytes)],
          ["isdir", RTV.logical(e.isdir)],
          ["datenum", RTV.num(datenum)],
        ])
      );
    });

    return RTV.structArray(fieldNames, elements);
  },
});

builtin({
  name: "tempdir",
  nargin: 0,
  mfile: true,
  apply: (_args, _nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.tempdir)
      throw new RuntimeError("tempdir is not available in this environment");
    return RTV.char(io.tempdir());
  },
});

builtin({
  name: "userpath",
  nargin: [0, 1],
  mfile: true,
  apply: (_args, _nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.userpath)
      throw new RuntimeError("userpath is not available in this environment");
    return RTV.char(io.userpath());
  },
});

builtin({
  name: "tempname",
  nargin: [0, 1],
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    let folder: string;
    if (args.length >= 1) {
      folder = toString(args[0]);
    } else {
      if (!io.tempdir)
        throw new RuntimeError("tempname is not available in this environment");
      folder = io.tempdir();
    }
    const name = "tp" + Math.random().toString(36).slice(2, 18);
    return RTV.char(folder + "/" + name);
  },
});
