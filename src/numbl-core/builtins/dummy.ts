/**
 * Dummy no-op built-in functions that return a dummy_handle.
 * Used as placeholders for unsupported functionality.
 */

import { register, builtinSingle } from "./registry.js";
import { IType } from "../lowering/itemTypes.js";
import { RTV } from "../runtime/index.js";

/** Register a group of names that all share the same dummy builtin. */
function registerDummyGroup(
  names: string[],
  fn: ReturnType<typeof builtinSingle>
): void {
  for (const name of names) register(name, fn);
}

const dummyHandleFunctions = [
  "groot",
  "gcf",
  "gca",
  "shg",
  "dir",
  "newplot",
  "caxis",
  "axis",
  "odeset",
];
const returnDummyStringFunctions = [
  "pwd",
  "datestr",
  "now",
  "lastwarn",
  "mfilename",
];
const returnEmptyArrayFunctions = ["xlim", "ylim"];
const returnDummyBooleanFunctions = ["ispc", "ismac", "isunix"];
const returnDummyCellArrayFunctions = ["listfonts"];

function registerPathFunctions(): void {
  // These are dummy registrations so the compiler generates proper call code.
  // Real implementations are injected via customBuiltins during mip load execution.
  const dummyStr = builtinSingle(() => RTV.string(""), {
    outputType: IType.String,
  });
  register("fileparts", dummyStr);
  register("fullfile", dummyStr);
  register(
    "addpath",
    builtinSingle(() => undefined, { outputType: IType.Void })
  );
}

// File I/O stubs — real implementations are injected via ExecOptions.fileIO
// and registered in specialBuiltins.ts at runtime.
const fileIONumFunctions = ["fopen", "fclose", "feof"];
const fileIOStringFunctions = ["fileread", "ferror"];
const fileIOUnknownFunctions = ["fgetl", "fgets"]; // return string or -1

function registerFileIOFunctions(): void {
  const numFn = builtinSingle(() => RTV.num(0), { outputType: IType.num() });
  for (const name of fileIONumFunctions) {
    register(name, numFn);
  }
  const strFn = builtinSingle(() => RTV.string(""), {
    outputType: IType.String,
  });
  for (const name of fileIOStringFunctions) {
    register(name, strFn);
  }
  const unknownFn = builtinSingle(() => RTV.num(0), {
    outputType: IType.Unknown,
  });
  for (const name of fileIOUnknownFunctions) {
    register(name, unknownFn);
  }
}

function registerHandleGetSet(): void {
  // get(handle, propName) — return a dummy handle for any property
  register(
    "get",
    builtinSingle(() => RTV.dummyHandle(), {
      outputType: IType.DummyHandle,
    })
  );

  // set(handle, ...) — void no-op (set returns nothing)
  register(
    "set",
    builtinSingle(() => undefined, {
      outputType: IType.Void,
    })
  );
}

export function getDummyBuiltinNames(): string[] {
  return [
    ...dummyHandleFunctions,
    ...returnDummyStringFunctions,
    ...returnEmptyArrayFunctions,
    ...returnDummyBooleanFunctions,
    ...returnDummyCellArrayFunctions,
    "get",
    "set",
    "fileparts",
    "fullfile",
    "addpath",
    ...fileIONumFunctions,
    ...fileIOStringFunctions,
    ...fileIOUnknownFunctions,
    "who",
    "whos",
  ];
}

export const registerDummyFunctions = () => {
  registerDummyGroup(
    dummyHandleFunctions,
    builtinSingle(() => RTV.dummyHandle(), { outputType: IType.DummyHandle })
  );
  registerDummyGroup(
    returnDummyStringFunctions,
    builtinSingle(() => RTV.string(""), { outputType: IType.String })
  );
  registerDummyGroup(
    returnEmptyArrayFunctions,
    builtinSingle(() => RTV.tensor([], [0, 0]), { outputType: IType.tensor() })
  );
  registerDummyGroup(
    returnDummyBooleanFunctions,
    builtinSingle(() => RTV.logical(false), { outputType: IType.Logical })
  );
  registerDummyGroup(
    returnDummyCellArrayFunctions,
    builtinSingle(() => RTV.cell([], [0, 0]), {
      outputType: IType.cell(IType.Unknown, "unknown"),
    })
  );
  registerHandleGetSet();
  registerPathFunctions();
  registerFileIOFunctions();
  // who()/whos() are handled as compile-time intrinsics (codegenExpr) + runtime methods,
  // but need builtin registrations so the compiler recognizes them as functions.
  register(
    "who",
    builtinSingle(() => RTV.cell([], [0, 0]), {
      outputType: IType.cell(IType.Char),
    })
  );
  register(
    "whos",
    builtinSingle(() => RTV.struct(new Map()), {
      outputType: IType.Unknown,
    })
  );
};
