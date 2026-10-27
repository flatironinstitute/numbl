import { describe, it, expect } from "vitest";
import {
  decodeShareData,
  encodeShareData,
  makeShareHash,
} from "../utils/shareUrl";

// Produced by the former pako-based encoder; existing share links must keep
// decoding.
const LEGACY_HASH =
  "eJyrVkrLzEktVrKKrlbKS8xNVbJSyk3MzNPLVdJRSs7PK0nNKwEKVSjYKuRk5hUXJCanahjoKBjqKJhqWsfkpWQWF2gUl-ZqVGhqxuSpKhzeo_CoY2JMnlKtDty4nMwk_YzUnILUIjRD00rzkksy8_MUKoGmQ1QAzYnJUwALGGlVAC1IzUsBmRaro5QIVFuW6gZ0rB-KM2sBWQlAWw";

const LEGACY_DATA = {
  files: [
    {
      name: "main.m",
      content: "x = linspace(0, 1, 5);\ndisp(sum(x))\n% ü ∑\n",
    },
    {
      name: "lib/helper.m",
      content: "function y = helper(x)\n  y = 2*x;\nend\n",
    },
  ],
  activeFileName: "main.m",
};

describe("shareUrl", () => {
  it("decodes a hash written by the previous encoder", () => {
    expect(decodeShareData(LEGACY_HASH)).toEqual(LEGACY_DATA);
  });

  it("round-trips project files", () => {
    const enc = new TextEncoder();
    const files = LEGACY_DATA.files.map((f, i) => ({
      id: String(i),
      name: f.name,
    }));
    const contents = new Map(
      LEGACY_DATA.files.map((f, i) => [String(i), enc.encode(f.content)])
    );
    const hash = encodeShareData(files, contents, "0");
    expect(hash).toBe(LEGACY_HASH);
    expect(decodeShareData(hash)).toEqual(LEGACY_DATA);
  });

  it("round-trips a single script", () => {
    const hash = makeShareHash("demo", "disp(1)\n");
    expect(decodeShareData(hash)).toEqual({
      files: [{ name: "demo.m", content: "disp(1)\n" }],
      activeFileName: "demo.m",
    });
  });
});
