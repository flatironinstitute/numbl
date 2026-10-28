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
} from "../../runtime/types.js";
import { sprintfFormat } from "../../helpers/string.js";
import { getAllBuiltinNames } from "../../helpers/registry.js";
import { getIBuiltinHelp } from "./types.js";
import { builtin } from "./builtin.js";

builtin({
  name: "help",
  mfile: true,
  apply: (args, nargout, ctx) => {
    let text = "";
    const emit = (s: string) => {
      text += s;
      if (nargout === 0) ctx.rt.output(s);
    };
    if (args.length === 0) {
      const names = getAllBuiltinNames().sort();
      emit("Available builtins:\n");
      emit("  " + names.join(", ") + "\n");
      emit("\nType 'help <name>' for help on a specific builtin.\n");
    } else {
      const name = toString(args[0]);
      const h = getIBuiltinHelp(name);
      if (!h) {
        const allNames = getAllBuiltinNames();
        if (allNames.includes(name)) {
          emit(`No help available for '${name}'.\n`);
        } else {
          emit(`Unknown function '${name}'.\n`);
        }
      } else {
        emit(`  ${h.signatures.join("\n  ")}\n\n`);
        emit(`${h.description}\n`);
      }
    }
    return nargout >= 1 ? RTV.char(text) : undefined;
  },
});

builtin({
  name: "disp",
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
    // warning('on'/'off') — global toggle
    if (args.length === 1 && isRuntimeChar(args[0])) {
      const mode = toString(args[0]);
      if (mode === "on" || mode === "off") {
        const prev = stateStruct("all");
        setState(mode, "all");
        return nargout >= 1 ? prev : undefined;
      }
    }
    // warning('on'/'off'/'query', id) — state query/set form
    if (args.length === 2 && isRuntimeChar(args[0]) && isRuntimeChar(args[1])) {
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
      isRuntimeChar(args[0]) &&
      toString(args[0]).includes(":")
    ) {
      fmtIdx = 1;
    }
    // Suppress output for disabled warning ids
    if (isOff(fmtIdx === 1 ? toString(args[0]) : "all")) {
      return nargout >= 1 ? RTV.num(0) : undefined;
    }
    const fmt = toString(args[fmtIdx]);
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
    if (fmtArgs.length === 0) {
      ctx.rt.output("Warning: " + fmt + "\n");
    } else {
      ctx.rt.output("Warning: " + sprintfFormat(fmt, fmtArgs) + "\n");
    }
    return nargout >= 1 ? RTV.num(0) : undefined;
  },
});

builtin({
  name: "fprintf",
  nargin: [1, Infinity],
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
      rt.output(output);
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
