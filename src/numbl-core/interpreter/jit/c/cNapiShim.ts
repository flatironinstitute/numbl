/**
 * Generates a pure-C Node-API (`<node_api.h>`) wrapper for a C-JIT'd
 * function so it can be dlopened and invoked from JavaScript.
 *
 * Staying in pure C avoids the C++ / `node-addon-api` header dependency,
 * which keeps the compile invocation as simple as
 *   cc -O2 -shared -fPIC -I<headers> src.c -lm -o out.node
 *
 * Arg marshalling:
 *   - scalar (number/boolean) → double. `napi_coerce_to_number` first so
 *     JS booleans round-trip correctly.
 *   - tensor (RuntimeTensor) → napi_value, passed through unchanged. The
 *     C function side deconstructs it via the helpers in cJitHelpers.ts.
 *
 * Return marshalling:
 *   - number  → napi_create_double
 *   - boolean → napi_get_boolean (preserves MATLAB `logical` class)
 *   - tensor  → returned napi_value passed through (already a RuntimeTensor)
 *
 * When the underlying C function touches tensors, it takes `napi_env env`
 * as its first parameter; the shim threads env from `napi_callback_info`.
 */

import type { CParamDesc } from "./jitCodegenC.js";

export type ScalarCKind = "number" | "boolean";
export type ReturnCKind = "number" | "boolean" | "tensor";

export interface NapiShimResult {
  /** C source for the shim (include directives + functions). */
  shim: string;
  /** JavaScript property name the loaded module exposes the function under. */
  exportName: string;
}

export function generateNapiShim(
  cFnName: string,
  paramDescs: CParamDesc[],
  returnKind: ReturnCKind,
  usesTensors: boolean
): NapiShimResult {
  const exportName = "fn";
  const nArgs = paramDescs.length;
  const isTensorReturn = returnKind === "tensor";

  // Build the extern declaration. Must match generateC's emitted signature:
  //   [napi_env env,] (double | napi_value)..., → (double | napi_value)
  const externParts: string[] = [];
  if (usesTensors) externParts.push("napi_env");
  for (const p of paramDescs) {
    externParts.push(p.kind === "tensor" ? "napi_value" : "double");
  }
  const externSig = externParts.length === 0 ? "void" : externParts.join(", ");
  const externRet = isTensorReturn ? "napi_value" : "double";

  // Build the per-arg local declarations + N-API extracts + the call-arg
  // names that feed into the extern invocation.
  const argDecls: string[] = [];
  const argExtracts: string[] = [];
  const callArgs: string[] = [];
  if (usesTensors) callArgs.push("env");

  for (let i = 0; i < nArgs; i++) {
    const p = paramDescs[i];
    const safeName = p.name.replace(/[^A-Za-z0-9_]/g, "_");
    if (p.kind === "scalar") {
      const local = `s_${safeName}_${i}`;
      argDecls.push(`  double ${local} = 0.0;`);
      argExtracts.push(
        `  {\n` +
          `    napi_value n;\n` +
          `    status = napi_coerce_to_number(env, argv[${i}], &n);\n` +
          `    if (status != napi_ok) return NULL;\n` +
          `    status = napi_get_value_double(env, n, &${local});\n` +
          `    if (status != napi_ok) return NULL;\n` +
          `  }`
      );
      callArgs.push(local);
    } else {
      // Tensor arg: pass the JS value straight through.
      callArgs.push(`argv[${i}]`);
    }
  }

  // Wrap the return value for the JS side.
  let resultDecl: string;
  let returnWrap: string;
  if (returnKind === "tensor") {
    resultDecl = `  napi_value result = ${cFnName}(${callArgs.join(", ")});`;
    returnWrap =
      `  /* A NULL return means the C function already threw via N-API\n` +
      `   * (e.g. the bail-sentinel path). Propagate by returning NULL. */\n` +
      `  if (result == NULL) return NULL;\n` +
      `  return result;`;
  } else if (returnKind === "boolean") {
    resultDecl = `  double result = ${cFnName}(${callArgs.join(", ")});`;
    returnWrap =
      `  napi_value ret;\n` +
      `  status = napi_get_boolean(env, result != 0.0, &ret);\n` +
      `  if (status != napi_ok) return NULL;\n` +
      `  return ret;`;
  } else {
    resultDecl = `  double result = ${cFnName}(${callArgs.join(", ")});`;
    returnWrap =
      `  napi_value ret;\n` +
      `  status = napi_create_double(env, result, &ret);\n` +
      `  if (status != napi_ok) return NULL;\n` +
      `  return ret;`;
  }

  // When the C function may have thrown during a scalar return path
  // (via numbl_jit_bail inside a reduction helper, for instance), we
  // need to propagate that exception rather than wrapping `result`
  // (which is 0.0 after a throw). Detect via napi_is_exception_pending.
  const throwGuard = usesTensors
    ? `  bool __pending = false;\n` +
      `  napi_is_exception_pending(env, &__pending);\n` +
      `  if (__pending) return NULL;\n`
    : "";

  const shim = `
/* N-API shim for ${cFnName} */
#include <node_api.h>
#include <stdbool.h>

extern ${externRet} ${cFnName}(${externSig});

static napi_value ${cFnName}_napi(napi_env env, napi_callback_info info) {
  napi_status status;
  size_t argc = ${nArgs};
${
  nArgs > 0
    ? `  napi_value argv[${nArgs}];
  status = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  if (status != napi_ok) return NULL;`
    : `  status = napi_get_cb_info(env, info, &argc, NULL, NULL, NULL);
  if (status != napi_ok) return NULL;`
}
${argDecls.join("\n")}
${argExtracts.join("\n")}
${resultDecl}
${throwGuard}${returnWrap}
}

NAPI_MODULE_INIT(/* env, exports */) {
  napi_status status;
  napi_value fn;
  status = napi_create_function(env, NULL, 0, ${cFnName}_napi, NULL, &fn);
  if (status != napi_ok) return NULL;
  status = napi_set_named_property(env, exports, ${JSON.stringify(
    exportName
  )}, fn);
  if (status != napi_ok) return NULL;
  return exports;
}
`.trim();

  return { shim, exportName };
}
