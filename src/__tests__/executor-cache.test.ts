import { describe, it, expect } from "vitest";
import {
  ExecutorCache,
  MAX_RUNTIME_BAILS,
} from "../numbl-core/executors/cache.js";

describe("ExecutorCache runtime bails", () => {
  it("stops after repeated runtime bails of a unit that never succeeded", () => {
    const cache = new ExecutorCache();
    const owner = {};
    for (let i = 0; i < MAX_RUNTIME_BAILS; i++) {
      expect(cache.keepsBailing("jit-loop", owner)).toBe(false);
      cache.noteRun("jit-loop", owner, true);
    }
    expect(cache.keepsBailing("jit-loop", owner)).toBe(true);
    // Per executor and per owner.
    expect(cache.keepsBailing("cjit-loop", owner)).toBe(false);
    expect(cache.keepsBailing("jit-loop", {})).toBe(false);
  });

  it("keeps compiling a unit that has run to completion", () => {
    const cache = new ExecutorCache();
    const owner = {};
    cache.noteRun("jit-call", owner, true);
    cache.noteRun("jit-call", owner, false);
    for (let i = 0; i < MAX_RUNTIME_BAILS + 2; i++)
      cache.noteRun("jit-call", owner, true);
    expect(cache.keepsBailing("jit-call", owner)).toBe(false);
  });
});
