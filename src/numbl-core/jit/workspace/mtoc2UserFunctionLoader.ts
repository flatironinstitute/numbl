/**
 * Loader for `.mtoc2.js` user functions.
 *
 * A `.mtoc2.js` file is plain JavaScript with no mtoc2 dependency.
 * The loader evaluates it as a CommonJS-style module (`exports` and
 * `module.exports` are the only injected names) and reads four
 * exports to assemble a workspace-scoped `Builtin`:
 *
 *   exports.name      — string, must match the file's basename.
 *   exports.transfer  — (argTypes, nargout) -> Type[]. Raw mtoc2 Type
 *                       objects in, raw Type objects out. Throws
 *                       `Error` on bad input.
 *   exports.emit      — ({argsC, prefix, ...}) -> string. C expression
 *                       for the call. Use `prefix` to name any private
 *                       C helpers (kept collision-free across files).
 *   exports.cBody     — ({prefix}) -> string. C source (helper
 *                       function definitions) injected once per
 *                       translation unit. Use the same `prefix` as
 *                       `emit`.
 *   exports.cHeaders? — string[] of additional `#include` lines to
 *                       activate (optional).
 *
 * Prefix shape: `mtoc2_user_<safe_funcname>__<8hex>__` where the hex
 * is the FNV-1a hash of the workspace-relative file path. Including
 * the path in the hash makes the prefix unique across all `.mtoc2.js`
 * files in the workspace, even if two files happen to define
 * functions with overlapping internal names.
 */

import type { Builtin, InlineSnippet } from "../builtins/registry.js";
import { UnsupportedConstruct } from "../lowering/errors.js";
import { hashType } from "../lowering/types.js";

interface UserExports {
  name: string;
  transfer: (argTypes: unknown[], nargout: number) => unknown[];
  emit: (args: {
    argsC: string[];
    argTypes: unknown[];
    nargout: number;
    outArgsC?: string[];
    prefix: string;
  }) => string;
  cBody: (args: { prefix: string }) => string;
  cHeaders?: ReadonlyArray<string>;
  /** Sibling files (relative to the .mtoc2.js) that mtoc2 makes
   *  available to the C build. `.c` files are compiled along with
   *  the main translation unit; `.h` files just sit in the include
   *  path so the user's `.c` and the inline `cBody` can `#include`
   *  them. Paths are resolved relative to the `.mtoc2.js`'s
   *  directory. */
  cSources?: ReadonlyArray<string>;
  /** mtoc2 runtime snippet names that `cBody` references (e.g.
   *  `"mtoc2_tensor_alloc_nd"` when the wrapper allocates an output
   *  tensor). Each name is activated alongside the user's cBody so
   *  the snippet's definition lands in the emitted C. */
  runtimeDeps?: ReadonlyArray<string>;
}

/** Resolved sibling file: the relative path (used by the build to
 *  pick a unique on-disk name) plus the file's text. */
export interface ResolvedCSource {
  /** Path as written by the user in `exports.cSources` (e.g.
   *  `"impl.c"` or `"sub/helpers.h"`). */
  relPath: string;
  /** File text the build will write to disk. */
  source: string;
}

/** Loader output: the assembled `Builtin` plus the user-supplied C
 *  sibling files the framework needs to thread through to the build
 *  pipeline. */
export interface LoadedUserFunction {
  builtin: Builtin;
  cSources: ResolvedCSource[];
}

/** C-identifier-safe transform of a workspace function name. Dots in
 *  qualified names (`pkg.foo`) become underscores; anything else
 *  non-alphanumeric becomes underscore. */
function safeIdent(name: string): string {
  return name.replace(/[^A-Za-z0-9_]/g, "_");
}

/** Compute the per-file C-namespace prefix. Combines the function's
 *  qualified name (for readability in the emitted C) with a hash of
 *  the workspace-relative file path (for uniqueness across files).
 *  Same prefix on every load — the hash is deterministic. */
function computePrefix(funcName: string, relPath: string): string {
  return `mtoc2_user_${safeIdent(funcName)}__${hashType(relPath)}__`;
}

/** Evaluate `.mtoc2.js` source as a CommonJS-style module. The user
 *  may either set fields on `exports` (`exports.foo = ...`) or
 *  reassign `module.exports = {...}` — both produce the same final
 *  exports object. */
function evalSource(fileName: string, source: string): Record<string, unknown> {
  let factory: (module: { exports: unknown }, exports: unknown) => void;
  try {
    factory = new Function(
      "module",
      "exports",
      source + `\n//# sourceURL=${fileName}\n`
    ) as (module: { exports: unknown }, exports: unknown) => void;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': failed to parse — ${msg}`
    );
  }
  const exports = {};
  const module = { exports };
  try {
    factory(module, exports);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': error during evaluation — ${msg}`
    );
  }
  const out = module.exports;
  if (out === undefined || out === null || typeof out !== "object") {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': module.exports must be an object ` +
        `(got ${out === undefined ? "undefined" : out === null ? "null" : typeof out})`
    );
  }
  return out as Record<string, unknown>;
}

/** Validate the user's exports satisfy the contract. Throws
 *  `UnsupportedConstruct` (with the file name) on any mismatch. */
function validateExports(
  fileName: string,
  raw: Record<string, unknown>
): UserExports {
  const requireFn = (
    field: keyof UserExports
  ): UserExports[keyof UserExports] => {
    const v = raw[field];
    if (typeof v !== "function") {
      throw new UnsupportedConstruct(
        `mtoc2 user function '${fileName}': missing or non-function ` +
          `'${String(field)}' export`
      );
    }
    return v as UserExports[keyof UserExports];
  };
  if (typeof raw.name !== "string") {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': missing or non-string 'name' export`
    );
  }
  const headers = raw.cHeaders;
  if (
    headers !== undefined &&
    !(Array.isArray(headers) && headers.every(h => typeof h === "string"))
  ) {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': 'cHeaders' must be a string[] when present`
    );
  }
  const cSources = raw.cSources;
  if (
    cSources !== undefined &&
    !(Array.isArray(cSources) && cSources.every(s => typeof s === "string"))
  ) {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': 'cSources' must be a string[] when present`
    );
  }
  const runtimeDeps = raw.runtimeDeps;
  if (
    runtimeDeps !== undefined &&
    !(
      Array.isArray(runtimeDeps) &&
      runtimeDeps.every(s => typeof s === "string")
    )
  ) {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': 'runtimeDeps' must be a string[] when present`
    );
  }
  return {
    name: raw.name,
    transfer: requireFn("transfer") as UserExports["transfer"],
    emit: requireFn("emit") as UserExports["emit"],
    cBody: requireFn("cBody") as UserExports["cBody"],
    ...(headers !== undefined ? { cHeaders: headers as string[] } : {}),
    ...(cSources !== undefined ? { cSources: cSources as string[] } : {}),
    ...(runtimeDeps !== undefined
      ? { runtimeDeps: runtimeDeps as string[] }
      : {}),
  };
}

/** Evaluate a `.mtoc2.js` file and return its assembled `Builtin`
 *  plus the user-supplied sibling C files (`cSources`).
 *
 *  - `fileName` — the .mtoc2.js's full path (used for diagnostics and
 *    to resolve sibling paths).
 *  - `expectedName` — what numbl's workspace registered the file
 *    under (derived from the file's basename / package path); we
 *    validate the user's `exports.name` matches so a typo / rename
 *    surfaces here rather than at the first call site.
 *  - `relPath` — workspace-relative path used for the prefix hash.
 *  - `readSibling(p)` — given a path-key relative to the .mtoc2.js's
 *    directory, return the sibling file's source text. Used to pull
 *    `cSources` entries out of the workspace's file map (which is
 *    the same regardless of CLI vs web IDE). Returns `undefined` if
 *    the file is not in the workspace; the loader throws a clear
 *    error pointing at the missing path. */
export function loadMtoc2UserFunction(
  fileName: string,
  source: string,
  expectedName: string,
  relPath: string,
  readSibling: (siblingRelPath: string) => string | undefined
): LoadedUserFunction {
  const raw = evalSource(fileName, source);
  const userExports = validateExports(fileName, raw);
  if (userExports.name !== expectedName) {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': 'name' export is ` +
        `'${userExports.name}' but the workspace registers this file ` +
        `as '${expectedName}' (rename the file or change the name export ` +
        `to match)`
    );
  }
  const prefix = computePrefix(expectedName, relPath);

  // Pre-run cBody once to capture the C source. Errors at this stage
  // surface with the file name attached so the user can fix typos at
  // workspace-init time, not at first call site.
  let cBodyText: string;
  try {
    cBodyText = userExports.cBody({ prefix });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': error evaluating cBody — ${msg}`
    );
  }
  if (typeof cBodyText !== "string") {
    throw new UnsupportedConstruct(
      `mtoc2 user function '${fileName}': cBody must return a string ` +
        `(got ${typeof cBodyText})`
    );
  }

  // Per-translation-unit snippet name — combines the user's function
  // name with the prefix hash so two files declaring same-name
  // functions in different locations stay distinct.
  const snippetName = `${prefix}_snippet`;
  const snippet: InlineSnippet = {
    name: snippetName,
    code: cBodyText,
    headers: userExports.cHeaders,
    // Runtime deps the user's cBody references (e.g.
    // `mtoc2_tensor_alloc_nd` for the common alloc-and-return
    // pattern). Activation pulls each in along with the cBody so
    // the definitions land in the emitted C.
    deps: userExports.runtimeDeps,
  };

  // Resolve every `cSources` entry against the workspace's file map.
  // A missing file is a clear-attribution error: the user wrote a
  // path that we can't find on disk / in the workspace.
  const resolvedCSources: ResolvedCSource[] = [];
  if (userExports.cSources !== undefined) {
    for (const rel of userExports.cSources) {
      if (typeof rel !== "string") {
        throw new UnsupportedConstruct(
          `mtoc2 user function '${fileName}': every 'cSources' entry must ` +
            `be a string path`
        );
      }
      const src = readSibling(rel);
      if (src === undefined) {
        throw new UnsupportedConstruct(
          `mtoc2 user function '${fileName}': 'cSources' entry '${rel}' ` +
            `not found in the workspace (sibling path resolved relative ` +
            `to the .mtoc2.js's directory)`
        );
      }
      resolvedCSources.push({ relPath: rel, source: src });
    }
  }

  const builtin: Builtin = {
    name: userExports.name,
    transfer(argTypes, nargout) {
      // Pass mtoc2's Type[] straight through to the user; they read
      // and return raw Type objects per the documented schema.
      const out = userExports.transfer(argTypes as unknown[], nargout);
      if (!Array.isArray(out)) {
        throw new UnsupportedConstruct(
          `mtoc2 user function '${expectedName}': transfer must return ` +
            `an array of Type, got ${typeof out}`
        );
      }
      return out as Builtin extends { transfer: (...a: never) => infer R }
        ? R
        : never;
    },
    emitC({ argsC, argTypes, nargout, outArgsC, useRuntime }) {
      useRuntime(snippet);
      const out = userExports.emit({
        argsC,
        argTypes: argTypes as unknown[],
        nargout,
        ...(outArgsC !== undefined ? { outArgsC } : {}),
        prefix,
      });
      if (typeof out !== "string") {
        throw new UnsupportedConstruct(
          `mtoc2 user function '${expectedName}': emit must return ` +
            `a string, got ${typeof out}`
        );
      }
      return out;
    },
  };

  return { builtin, cSources: resolvedCSources };
}
