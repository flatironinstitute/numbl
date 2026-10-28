import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { generateBuiltinNames } from "../../scripts/gen_builtin_names.js";

describe("src/builtinNames.gen.ts", () => {
  it("matches the builtin registry (run `npm run gen:builtin-names`)", () => {
    const path = resolve(import.meta.dirname, "../builtinNames.gen.ts");
    expect(readFileSync(path, "utf-8")).toBe(generateBuiltinNames());
  });
});
