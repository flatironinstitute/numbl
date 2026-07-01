/* Midpoint-rule quadrature of a numbl function handle.
 *
 * `fid` is a callback id registered on the JS side (`wasm.callbacks.add`).
 * Each evaluation calls back into the numbl handle via the host-provided
 * import `numbl_cb_d(id, x)`.
 *
 * For wasm: emcc myquad.c -O2 -s STANDALONE_WASM --no-entry -o myquad.wasm
 */

#ifdef __EMSCRIPTEN__
#define EXPORT __attribute__((export_name("myquad")))
#else
#define EXPORT
#endif

/* Host import: invoke the numbl handle with id `id` at point `x`. */
extern double numbl_cb_d(int id, double x);

EXPORT
double myquad(int fid, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += numbl_cb_d(fid, x);
    }
    return sum * h;
}
