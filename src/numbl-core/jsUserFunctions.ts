/**
 * Loader for .js user functions.
 *
 * Evaluates .js files that define branches via register({ check, apply }).
 */

import type { BuiltinFn, BuiltinFnBranch } from "./builtins/registry.js";
import { type ItemType, IType } from "./lowering/itemTypes.js";
import type { WorkspaceFile } from "./workspace/index.js";
import { RTV, RuntimeError } from "./runtime/index.js";
import { FloatXArray } from "./runtime/types.js";

/**
 * Derive a function name from a .js workspace file path.
 * E.g., "myadd.js" → "myadd", "/path/to/myadd.js" → "myadd"
 */
function funcNameFromFile(fileName: string): string {
  const base = fileName.split("/").pop()!;
  return base.replace(/\.js$/, "");
}

const defaultCheck = (_argTypes: ItemType[], nargout: number) => ({
  outputTypes: Array(Math.max(nargout, 1)).fill({
    kind: "Unknown",
  } as ItemType),
});

/**
 * Load .js user function files and return them as a map of function name → BuiltinFn.
 */
export function loadJsUserFunctions(
  jsFiles: WorkspaceFile[]
): Map<string, BuiltinFn> {
  const result = new Map<string, BuiltinFn>();

  for (const file of jsFiles) {
    const funcName = funcNameFromFile(file.name);
    try {
      const branches: BuiltinFnBranch[] = [];

      const registerFn = (branch: {
        check?: BuiltinFnBranch["check"];
        apply: BuiltinFnBranch["apply"];
      }) => {
        if (typeof branch.apply !== "function") {
          throw new Error("register(): branch must have an apply function");
        }
        branches.push({
          check: branch.check ?? defaultCheck,
          apply: branch.apply,
        });
      };

      const factory = new Function(
        "RTV",
        "RuntimeError",
        "FloatXArray",
        "IType",
        "register",
        file.source
      );
      factory(RTV, RuntimeError, FloatXArray, IType, registerFn);

      if (branches.length === 0) {
        throw new Error(
          `JS user function '${funcName}' (${file.name}) must call register() at least once`
        );
      }

      result.set(funcName, branches);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(
        `Error loading JS user function '${funcName}' (${file.name}): ${msg}`
      );
    }
  }

  return result;
}
