/* mtoc runtime helper: format a double the same way numbl's
 * `formatNumber` does, into a caller-provided buffer.
 *
 * Used by both scalar and tensor disp so cross-runner output matches
 * numbl byte-for-byte. Kept in its own snippet so disp_double.h and
 * disp_tensor.h can pull it in via the snippet dependency mechanism.
 *
 * Returns the number of characters written (excluding trailing NUL).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int mtoc2_format_double(char *out, size_t cap, double x) {
  if (isnan(x)) return snprintf(out, cap, "NaN");
  if (isinf(x)) return snprintf(out, cap, x > 0 ? "Infinity" : "-Infinity");
  if (x == 0.0) x = 0.0; /* normalize -0 */

  if (fabs(x) < 1e15 && x == (double)(long long)x) {
    return snprintf(out, cap, "%lld", (long long)x);
  }

  char buf[64];
  snprintf(buf, sizeof(buf), "%.4e", x);
  char *e_p = strchr(buf, 'e');
  if (!e_p) return snprintf(out, cap, "%s", buf);
  int dec_exp = atoi(e_p + 1);

  if (dec_exp < -6 || dec_exp >= 5) {
    /* Scientific. Strip trailing zeros from mantissa, drop leading
       zeros from the exponent, but keep its sign. */
    char *m_end = e_p - 1;
    while (m_end > buf && *m_end == '0') m_end--;
    if (m_end >= buf && *m_end == '.') m_end--;
    char *exp_p = e_p + 1;
    char sign = '+';
    if (*exp_p == '+' || *exp_p == '-') { sign = *exp_p; exp_p++; }
    while (*exp_p == '0' && *(exp_p + 1) != '\0') exp_p++;
    return snprintf(out, cap, "%.*se%c%s",
                    (int)(m_end - buf + 1), buf, sign, exp_p);
  }

  int frac_digits = 4 - dec_exp;
  if (frac_digits < 0) frac_digits = 0;
  snprintf(buf, sizeof(buf), "%.*f", frac_digits, x);
  if (strchr(buf, '.')) {
    size_t len = strlen(buf);
    while (len > 0 && buf[len - 1] == '0') len--;
    if (len > 0 && buf[len - 1] == '.') len--;
    buf[len] = '\0';
  }
  return snprintf(out, cap, "%s", buf);
}
