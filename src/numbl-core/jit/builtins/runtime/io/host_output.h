/* @snippet name=jit_host_output */
/* JIT host output hook.
 *
 * When numbl runs an emitted spec as a JIT'd `.so`, it binds a writer
 * via `jit_set_host_write` so all output routes through numbl's
 * output stream (`rt.output`), keeping --opt 2 output captured the
 * same as the interpreter / JS-JIT paths.
 *
 * `jit_set_host_write` has external linkage so koffi can resolve it
 * after dlopen; everything else is internal. When no writer is bound
 * `jit_stdout` falls back to libc stdout.
 */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static void (*jit_host_write)(const char *, long) = 0;

void jit_set_host_write(void (*fn)(const char *, long)) {
  jit_host_write = fn;
}

static inline void jit_stdout(const char *bytes, long len) {
  if (len <= 0) return;
  if (jit_host_write) jit_host_write(bytes, len);
  else fwrite(bytes, 1, (size_t)len, stdout);
}

static inline void jit_stdout_s(const char *s) {
  jit_stdout(s, (long)strlen(s));
}

static inline void jit_stdout_printf(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  long len = n < (int)sizeof(buf) ? (long)n : (long)sizeof(buf) - 1;
  jit_stdout(buf, len);
}
