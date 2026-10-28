import { describe, it, expect } from "vitest";
// Loading the interpreter registers every builtin.
import "../numbl-core/executeCode.js";
import { getIBuiltin } from "../numbl-core/runtime/builtinRegistry.js";
import { HELP_TEXT } from "../numbl-core/interpreter/builtins/help-text.js";

describe("help-text.ts", () => {
  const names = Object.keys(HELP_TEXT);

  it("only has entries for registered builtins", () => {
    expect(names.filter(n => !getIBuiltin(n))).toEqual([]);
  });

  it("has no entry for a builtin with inline help", () => {
    expect(names.filter(n => getIBuiltin(n)?.help)).toEqual([]);
  });
});
