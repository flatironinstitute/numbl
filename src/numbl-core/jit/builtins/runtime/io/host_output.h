/* JIT host output hook.
 *
 * Standalone / AOT builds write program output (disp, fprintf, …)
 * straight to stdout. When numbl runs an emitted spec as a JIT'd `.so`,
 * it binds a writer via `mtoc2_set_host_write` so all output routes
 * through numbl's output stream (`rt.output`) instead — keeping --opt 2
 * output captured the same as the interpreter / JS-JIT paths.
 *
 * `mtoc2_set_host_write` has external linkage so koffi can resolve it
 * after dlopen; everything else is internal. When no writer is bound
 * (AOT / standalone) `mtoc2_stdout` falls back to libc stdout.
 */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static void (*mtoc2_host_write)(const char *, long) = 0;

void mtoc2_set_host_write(void (*fn)(const char *, long)) {
  mtoc2_host_write = fn;
}

static inline void mtoc2_stdout(const char *bytes, long len) {
  if (len <= 0) return;
  if (mtoc2_host_write) mtoc2_host_write(bytes, len);
  else fwrite(bytes, 1, (size_t)len, stdout);
}

static inline void mtoc2_stdout_s(const char *s) {
  mtoc2_stdout(s, (long)strlen(s));
}

static inline void mtoc2_stdout_printf(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  long len = n < (int)sizeof(buf) ? (long)n : (long)sizeof(buf) - 1;
  mtoc2_stdout(buf, len);
}
