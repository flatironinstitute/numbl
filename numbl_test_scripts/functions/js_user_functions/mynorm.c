/* L2 norm of a double array.
 *
 * For wasm:   emcc mynorm.c -O2 -s STANDALONE_WASM --no-entry -o mynorm.wasm
 * For native: gcc -shared -fPIC -O2 -o mynorm.so mynorm.c -lm
 */

#include <math.h>

#ifdef __EMSCRIPTEN__
#define EXPORT __attribute__((export_name("mynorm")))
#define EXPORT_ALLOC __attribute__((export_name("alloc_doubles")))
#define EXPORT_FREE __attribute__((export_name("free_doubles")))
#else
#define EXPORT
#define EXPORT_ALLOC
#define EXPORT_FREE
#endif

EXPORT
double mynorm(int n, const double *x) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

/* Simple bump allocator for wasm — lets JS copy data into linear memory. */
static unsigned char heap[1 << 20]; /* 1 MB */
static int heap_offset = 0;

EXPORT_ALLOC
double *alloc_doubles(int n) {
    /* Align to 8 bytes */
    int aligned = (heap_offset + 7) & ~7;
    double *ptr = (double *)(heap + aligned);
    heap_offset = aligned + n * (int)sizeof(double);
    return ptr;
}

EXPORT_FREE
void free_doubles(void) {
    heap_offset = 0;
}
