/**
 * Drift detection: ensure TS op-code enums match the C op-code enums.
 *
 * Calls numbl_dump_op_codes() in the native addon and compares to the
 * values in src/numbl-core/ops/opCodes.ts. If this test fails, one side
 * was changed without updating the other.
 *
 * Skipped automatically when the native addon isn't built.
 */

import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";
import { OpRealBin, OpComplexBin } from "../numbl-core/ops/opCodes.js";

function loadAddon(): unknown | null {
  try {
    const req = createRequire(import.meta.url);
    return req("../../build/Release/numbl_addon.node");
  } catch {
    return null;
  }
}

function expectedDump(): string {
  return (
    `real_binary:ADD=${OpRealBin.ADD},SUB=${OpRealBin.SUB},MUL=${OpRealBin.MUL},DIV=${OpRealBin.DIV};` +
    `complex_binary:ADD=${OpComplexBin.ADD},SUB=${OpComplexBin.SUB},MUL=${OpComplexBin.MUL},DIV=${OpComplexBin.DIV};`
  );
}

describe("op-codes sync (TS ↔ C)", () => {
  const addon = loadAddon() as { tensorOpDumpCodes?: () => string } | null;

  it.skipIf(!addon || !addon.tensorOpDumpCodes)(
    "TS opCodes.ts matches numbl_dump_op_codes() in C",
    () => {
      const native = addon!.tensorOpDumpCodes!();
      expect(native).toBe(expectedDump());
    }
  );
});
