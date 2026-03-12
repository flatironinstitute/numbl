/**
 * Dummy no-op built-in functions that return a dummy_handle.
 * Used as placeholders for unsupported functionality.
 */

import { register, builtinSingle } from "./registry.js";
import { IType } from "../lowering/itemTypes.js";
import { RTV } from "../runtime/index.js";

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

function registerDummyHandleFunctions(): void {
  const fn = builtinSingle(() => RTV.dummyHandle(), {
    outputType: IType.DummyHandle,
  });
  for (const name of dummyHandleFunctions) {
    register(name, fn);
  }
}

const returnDummyStringFunctions = [
  "pwd",
  "datestr",
  "now",
  "lastwarn",
  "mfilename",
];

function registerDummyStringFunctions(): void {
  const fn = builtinSingle(() => RTV.string(""), {
    outputType: IType.String,
  });
  for (const name of returnDummyStringFunctions) {
    register(name, fn);
  }
}

const returnEmptyArrayFunctions = ["who", "xlim", "ylim"];

function registerDummyArrayFunctions(): void {
  const fn = builtinSingle(() => RTV.tensor([], [0, 0]), {
    outputType: IType.tensor(),
  });
  for (const name of returnEmptyArrayFunctions) {
    register(name, fn);
  }
}

const returnDummyBooleanFunctions = ["ispc", "ismac", "isunix"];

function registerDummyBooleanFunctions(): void {
  const fn = builtinSingle(() => RTV.logical(false), {
    outputType: IType.Logical,
  });
  for (const name of returnDummyBooleanFunctions) {
    register(name, fn);
  }
}

const returnDummyCellArrayFunctions = ["listfonts"];

function registerDummyCellArrayFunctions(): void {
  const fn = builtinSingle(() => RTV.cell([], [0, 0]), {
    outputType: IType.cell(IType.Unknown, "unknown"),
  });
  for (const name of returnDummyCellArrayFunctions) {
    register(name, fn);
  }
}

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
  ];
}

export const registerDummyFunctions = () => {
  registerDummyHandleFunctions();
  registerDummyStringFunctions();
  registerDummyArrayFunctions();
  registerDummyBooleanFunctions();
  registerDummyCellArrayFunctions();
  registerHandleGetSet();
  registerPathFunctions();
  registerFileIOFunctions();
};
