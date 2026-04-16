/**
 * Generates a pure-C Node-API (`<node_api.h>`) wrapper for a C-JIT'd
 * function so it can be dlopened and invoked from JavaScript.
 *
 * Staying in pure C avoids the C++ / `node-addon-api` header dependency,
 * which keeps the compile invocation as simple as
 *   cc -O2 -shared -fPIC -I<headers> src.c -lm -o out.node
 *
 * The underlying C function always uses `double` on both sides of the
 * call. This shim handles the JS-side marshalling:
 *
 *   - Args flagged as boolean come in as JS booleans; the shim uses
 *     `napi_coerce_to_number` first so either a JS number *or* a JS
 *     boolean round-trips correctly.
 *   - A boolean-typed return is wrapped via `napi_get_boolean` so the
 *     interpreter (and `islogical()`) see a real JS boolean rather than
 *     a plain 0/1 number. This preserves MATLAB's `logical` class across
 *     the FFI boundary.
 */

export type ScalarCKind = "number" | "boolean";

export interface NapiShimResult {
  /** C source for the shim (include directives + functions). */
  shim: string;
  /** JavaScript property name the loaded module exposes the function under. */
  exportName: string;
}

export function generateNapiShim(
  cFnName: string,
  argKinds: ScalarCKind[],
  returnKind: ScalarCKind
): NapiShimResult {
  const exportName = "fn";
  const nArgs = argKinds.length;

  const argDecls: string[] = [];
  const argExtracts: string[] = [];
  const argNames: string[] = [];
  for (let i = 0; i < nArgs; i++) {
    argDecls.push(`  double a${i} = 0.0;`);
    // Coerce any primitive to number first so JS booleans (true → 1,
    // false → 0) flow through cleanly. `napi_get_value_double` alone
    // throws on a boolean input.
    argExtracts.push(
      `  {\n` +
        `    napi_value n${i};\n` +
        `    status = napi_coerce_to_number(env, argv[${i}], &n${i});\n` +
        `    if (status != napi_ok) return NULL;\n` +
        `    status = napi_get_value_double(env, n${i}, &a${i});\n` +
        `    if (status != napi_ok) return NULL;\n` +
        `  }`
    );
    argNames.push(`a${i}`);
  }

  const returnWrap =
    returnKind === "boolean"
      ? `  napi_value ret;\n  status = napi_get_boolean(env, result != 0.0, &ret);\n  if (status != napi_ok) return NULL;\n  return ret;`
      : `  napi_value ret;\n  status = napi_create_double(env, result, &ret);\n  if (status != napi_ok) return NULL;\n  return ret;`;

  const shim = `
/* N-API shim for ${cFnName} */
#include <node_api.h>

extern double ${cFnName}(${
    nArgs === 0 ? "void" : Array(nArgs).fill("double").join(", ")
  });

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
  double result = ${cFnName}(${argNames.join(", ")});
${returnWrap}
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
