/* Shared library for mydot — compile with:
 *   gcc -shared -fPIC -O2 -o mydot.so mydot.c
 */
double mydot(int n, const double *a, const double *b) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
