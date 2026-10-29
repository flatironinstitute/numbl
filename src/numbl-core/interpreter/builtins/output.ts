/**
 * Output builtins that write to the runtime's output channel: help, disp,
 * warning, fprintf.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  toNumber,
  displayValue,
  RuntimeError,
} from "../../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeStringArray,
  isRuntimeString,
} from "../../runtime/types.js";
import { sprintfFormat } from "../../helpers/string.js";
import {
  getAllIBuiltinNames,
  getIBuiltinHelp,
  isBuiltin,
} from "../../runtime/builtinRegistry.js";
import { builtin } from "./builtin.js";
import { PRINTF_SPARSE } from "./args.js";

/** The leading help comment block of a .m source file: the comment lines
 *  right after the function/classdef declaration (or at the top of a
 *  script), up to the first non-comment line, with the `%` and one
 *  following space stripped. */
function extractMHelp(source: string): string {
  const lines = source.split(/\r?\n/);
  let i = 0;
  // Skip a leading declaration, including its continuation lines.
  if (i < lines.length && /^\s*(function|classdef)\b/.test(lines[i])) {
    while (i < lines.length && /\.\.\.\s*(%[^\n]*)?$/.test(lines[i])) i++;
    i++;
  }
  const helpLines: string[] = [];
  let sawComment = false;
  while (i < lines.length) {
    const trimmed = lines[i].trimStart();
    if (trimmed.startsWith("%")) {
      let line = trimmed.slice(1);
      if (line.startsWith(" ")) line = line.slice(1);
      helpLines.push(line);
      sawComment = true;
      i++;
    } else if (!sawComment && trimmed === "") {
      i++; // blank lines before the first comment
    } else {
      break;
    }
  }
  return helpLines.join("\n");
}

/** A path-style help target (`+mip/install.m`) as the function name numbl
 *  uses (`mip.install`); other names are returned unchanged. */
function helpPathToFuncName(name: string): string {
  if (!/[/\\]/.test(name) && !name.endsWith(".m")) return name;
  const parts = name
    .replace(/\.m$/, "")
    .split(/[/\\]/)
    .filter(p => p.length > 0);
  return parts.map(p => (p.startsWith("+") ? p.slice(1) : p)).join(".");
}

builtin({
  name: "help",
  mfile: true,
  scope: true,
  apply: (args, nargout, { rt, scope }) => {
    let text = "";
    const emit = (s: string) => {
      text += s;
      if (nargout === 0) rt.output(s);
    };
    const name = args.length > 0 ? toString(args[0]) : undefined;
    if (name === undefined) {
      const names = getAllIBuiltinNames().sort();
      emit("Available builtins:\n");
      emit("  " + names.join(", ") + "\n");
      emit("\nType 'help <name>' for help on a specific builtin.\n");
    } else {
      const h = getIBuiltinHelp(name);
      // A workspace .m function or class (or a .numbl.js function with
      // registered help): its leading comment block. Bare names
      // ("mip.install") and path-style targets ("+mip/install.m") work, as
      // in MATLAB.
      let fileHelp = "";
      if (!h && !isBuiltin(name)) {
        const funcName = helpPathToFuncName(name);
        const ws =
          scope.lookupWorkspaceFile(funcName) ??
          (funcName !== name ? scope.lookupWorkspaceFile(name) : undefined);
        if (ws)
          fileHelp =
            ws.kind === "jsfunction"
              ? ws.source.trimEnd()
              : extractMHelp(ws.source);
      }
      if (h) {
        emit(`  ${h.signatures.join("\n  ")}\n\n`);
        emit(`${h.description}\n`);
      } else if (fileHelp) {
        emit(fileHelp + "\n");
      } else if (isBuiltin(name)) {
        emit(`No help available for '${name}'.\n`);
      } else {
        emit(`Unknown function '${name}'.\n`);
      }
    }
    return nargout >= 1 ? RTV.char(text) : undefined;
  },
});

builtin({
  name: "disp",
  accepts: { sparse: true },
  nargin: 1,
  nargout: 0,
  apply: (args, _nargout, ctx) => {
    const v = args[0];
    if (isRuntimeTensor(v) && v.data.length === 0) return;
    ctx.rt.output(displayValue(v) + "\n");
  },
});

builtin({
  name: "warning",
  apply: (args, nargout, ctx) => {
    if (args.length === 0) return nargout >= 1 ? RTV.num(0) : undefined;
    const warningStateOff = ctx.rt.warningsOff;
    const isOff = (id: string) =>
      warningStateOff.has(id) || warningStateOff.has("all");
    const stateStruct = (id: string) =>
      RTV.struct(
        new Map<string, RuntimeValue>([
          ["identifier", RTV.char(id)],
          ["state", RTV.char(isOff(id) ? "off" : "on")],
        ])
      );
    const setState = (mode: string, id: string) => {
      if (mode === "off") warningStateOff.add(id);
      else {
        warningStateOff.delete(id);
        if (id === "all") warningStateOff.clear();
      }
    };
    // warning(s) — restore state from a struct previously returned by
    // warning('on'/'off'/'query', id)
    const restoreFrom = (s: RuntimeValue): boolean => {
      if (!isRuntimeStruct(s)) return false;
      const idField = s.fields.get("identifier");
      const stField = s.fields.get("state");
      if (idField === undefined || stField === undefined) return false;
      setState(toString(stField), toString(idField));
      return true;
    };
    if (args.length === 1 && isRuntimeStruct(args[0])) {
      if (restoreFrom(args[0])) return nargout >= 1 ? RTV.num(0) : undefined;
    }
    if (args.length === 1 && isRuntimeStructArray(args[0])) {
      for (const el of args[0].elements) restoreFrom(el);
      return nargout >= 1 ? RTV.num(0) : undefined;
    }
    // Char and string scalars are interchangeable as modes, ids and messages.
    const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);
    // warning('on'/'off') — global toggle
    if (args.length === 1 && isText(args[0])) {
      const mode = toString(args[0]);
      if (mode === "on" || mode === "off") {
        const prev = stateStruct("all");
        setState(mode, "all");
        return nargout >= 1 ? prev : undefined;
      }
    }
    // warning('on'/'off'/'query', id) — state query/set form
    if (args.length === 2 && isText(args[0]) && isText(args[1])) {
      const mode = toString(args[0]);
      const id = toString(args[1]);
      if (mode === "on" || mode === "off") {
        const prev = stateStruct(id);
        setState(mode, id);
        return nargout >= 1 ? prev : undefined;
      }
      if (mode === "query") {
        return stateStruct(id);
      }
    }
    // Detect warning(msgID, msg, ...) form — msgID contains a ':'
    let fmtIdx = 0;
    if (
      args.length >= 2 &&
      isText(args[0]) &&
      toString(args[0]).includes(":")
    ) {
      fmtIdx = 1;
    }
    // Suppress output for disabled warning ids
    if (isOff(fmtIdx === 1 ? toString(args[0]) : "all")) {
      return nargout >= 1 ? RTV.num(0) : undefined;
    }
    const fmt = toString(args[fmtIdx]);
    // As in MATLAB, a lone warning(msg) is taken literally; the id form
    // and any form with arguments format.
    if (args.length === 1) {
      ctx.rt.output("Warning: " + fmt + "\n");
      return nargout >= 1 ? RTV.num(0) : undefined;
    }
    const fmtArgs: RuntimeValue[] = [];
    for (let i = fmtIdx + 1; i < args.length; i++) {
      const a = args[i];
      if (isRuntimeTensor(a)) {
        for (let j = 0; j < a.data.length; j++)
          fmtArgs.push(RTV.num(a.data[j]));
      } else {
        fmtArgs.push(a);
      }
    }
    ctx.rt.output("Warning: " + sprintfFormat(fmt, fmtArgs) + "\n");
    return nargout >= 1 ? RTV.num(0) : undefined;
  },
});

builtin({
  name: "fprintf",
  nargin: [1, Infinity],
  accepts: { sparse: PRINTF_SPARSE },
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    // Detect optional leading fid argument (numeric first arg with >= 2 args)
    let fid = 1; // default stdout
    let fmtIdx = 0;
    if (args.length >= 2 && isRuntimeNumber(args[0])) {
      fid = toNumber(args[0]);
      fmtIdx = 1;
    }

    const fmt = toString(args[fmtIdx]);
    // sprintfFormat handles tensor flattening and format cycling internally,
    // so we can pass the remaining args through directly. String arrays
    // supply one text arg per element.
    const rest = args
      .slice(fmtIdx + 1)
      .flatMap(a =>
        isRuntimeStringArray(a) ? a.data.map(s => RTV.string(s)) : [a]
      );
    const output = sprintfFormat(fmt, rest);

    let count: number | undefined;
    if (fid === 1 || fid === 2) {
      rt.output(output, fid === 2);
    } else {
      if (!rt.files)
        throw new RuntimeError("File I/O is not available in this environment");
      count = rt.files.writeText(fid, output);
    }
    // MATLAB: called as a statement, fprintf does not set `ans`. Only
    // surface the byte count when the caller asks for it.
    return nargout >= 1 ? RTV.num(count ?? output.length) : undefined;
  },
});
