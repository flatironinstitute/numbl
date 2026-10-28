/* JS-only snippet (paired with `scalar_index.js`).
 *
 * The C path has its own bounds-checking helpers (`jit_idx_lin`,
 * `jit_idx_axis`) in `oob.h` plus inline column-major offset
 * arithmetic in `emit_index.ts`. This snippet exists only to give
 * the activation system a `.h` sibling for the JS-side helpers
 * (`jit_idx_lin_js`, `jit_idx_axis_js`); the C body is
 * intentionally empty.
 */
