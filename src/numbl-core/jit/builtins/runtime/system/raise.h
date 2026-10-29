/* @snippet name=jit_raise */
/* JIT runtime: the raise channel, statement locations, and the tracked
 * allocator. Always emitted first (emit.ts), so every later snippet and the
 * user code see the allocator macros below.
 *
 * Raise. Every runtime error or bail in compiled C calls `jit_raise(kind,
 * identifier, message)`, which records the status and `longjmp`s to the
 * guard in the exported host-entry wrapper (`<spec>__host`, emitted by
 * emit.ts). The wrapper frees what the call allocated and returns; the host
 * reads the status through the exported `jit_raise_*` accessors and maps it
 * (executors/jit/raise.ts):
 *   JIT_RAISE_BAIL   a JIT limitation, e.g. an indexed store that would grow
 *                    an array. The interpreter re-runs the unit and the unit
 *                    is cached as bailed.
 *   JIT_RAISE_ERROR  a MATLAB runtime error. `message` is the interpreter's
 *                    exact wording; NULL means the JIT does not reproduce it,
 *                    and the interpreter re-runs the unit to raise it.
 *   JIT_RAISE_CANCEL the host's cancel flag was set (the IDE's Stop). The
 *                    host raises numbl's CancellationError.
 *
 * Cancel. `jit__cancel` points at the host's cancel flag (a
 * SharedArrayBuffer word another thread sets, installed by the host through
 * `jit_set_cancel`), or at a constant zero. Emitted code polls it at the top
 * of every loop iteration (once per chunk of a strip-mined loop, see
 * codegen/cancelPoll.ts) and at the entry of functions that call other
 * specs (`JIT_POLL_CANCEL`): one relaxed load and a never-taken branch.
 *
 * Location. Emitted code stores the current source statement in `jit__loc`
 * (`file index << 32 | offset`) at the start of each statement, mirroring
 * the interpreter's per-statement `rt.$line`, so the host can attribute an
 * error to the right line.
 *
 * Memory. A raise abandons the frames between it and the guard, together
 * with whatever tensors they own. To release those, every allocation goes
 * through a tracked allocator: blocks carry a 16-byte header linking them
 * into one live list, `malloc` / `calloc` / `realloc` / `free` are macros
 * for the tracked versions, and the guard frees every live block. Between
 * calls the list is empty except for returned output buffers, which the
 * host releases with `jit_free`; input buffers the host allocates with
 * `jit_malloc` are owned (and freed) by the callee, and are freed by the
 * guard on a raise. The JIT runs one call at a time per `.so`, so the
 * file-scope state needs no locking.
 */

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JIT_RAISE_NONE 0
#define JIT_RAISE_BAIL 1
#define JIT_RAISE_ERROR 2
#define JIT_RAISE_CANCEL 3

/* ── Tracked allocator ─────────────────────────────────────────────── */

typedef struct jit__blk {
  struct jit__blk *prev;
  struct jit__blk *next;
} jit__blk_t;

static jit__blk_t jit__live = {&jit__live, &jit__live};

static void jit__blk_link(jit__blk_t *b) {
  b->prev = &jit__live;
  b->next = jit__live.next;
  jit__live.next->prev = b;
  jit__live.next = b;
}

static void jit__blk_unlink(jit__blk_t *b) {
  b->prev->next = b->next;
  b->next->prev = b->prev;
}

/* The header of the block holding `p`. Tensor buffer pointers are
 * `restrict` (tensor.h), so a header reached by arithmetic on one would
 * count as accessed through that restrict pointer, while linking a
 * neighbor writes the same header through another pointer. GCC then
 * keeps stale links in registers and the list (and the heap) breaks.
 * The empty asm hides the derivation from the optimizer. */
static jit__blk_t *jit__blk_of(void *p) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+r"(p));
#endif
  return (jit__blk_t *)p - 1;
}

static void *jit__tmalloc(size_t n) {
  jit__blk_t *b = (jit__blk_t *)(malloc)(sizeof(jit__blk_t) + (n > 0 ? n : 1));
  if (!b) return NULL;
  jit__blk_link(b);
  return (void *)(b + 1);
}

static void jit__tfree(void *p) {
  if (!p) return;
  jit__blk_t *b = jit__blk_of(p);
  jit__blk_unlink(b);
  (free)(b);
}

static void *jit__trealloc(void *p, size_t n) {
  if (!p) return jit__tmalloc(n);
  jit__blk_t *b = jit__blk_of(p);
  jit__blk_unlink(b);
  jit__blk_t *nb =
    (jit__blk_t *)(realloc)(b, sizeof(jit__blk_t) + (n > 0 ? n : 1));
  if (!nb) {
    jit__blk_link(b);
    return NULL;
  }
  jit__blk_link(nb);
  return (void *)(nb + 1);
}

static void *jit__tcalloc(size_t count, size_t size) {
  if (size != 0 && count > SIZE_MAX / size) return NULL;
  void *p = jit__tmalloc(count * size);
  if (p) memset(p, 0, count * size);
  return p;
}

/* Release every live block. Called by the host-entry guard after a raise. */
static void jit__free_all(void) {
  while (jit__live.next != &jit__live) {
    jit__blk_t *b = jit__live.next;
    jit__blk_unlink(b);
    (free)(b);
  }
}

#define malloc(n) jit__tmalloc(n)
#define calloc(c, s) jit__tcalloc((c), (s))
#define realloc(p, n) jit__trealloc((p), (n))
#define free(p) jit__tfree(p)

/* ── Statement location ────────────────────────────────────────────── */

static long long jit__loc = -1;

/* ── Raise ─────────────────────────────────────────────────────────── */

static jmp_buf jit__raise_buf;
static int jit__raise_kind = JIT_RAISE_NONE;
static long long jit__raise_loc = -1;
static char jit__raise_id[256];
/* Message (or bail reason) buffer. Grown with the untracked libc
 * allocator so the guard's `jit__free_all` leaves it alone. */
static char *jit__raise_msg = NULL;
static size_t jit__raise_msg_cap = 0;
static int jit__raise_has_msg = 0;

static void jit__raise_set_msg(const char *bytes, size_t len) {
  if (len + 1 > jit__raise_msg_cap) {
    size_t cap = len + 1 < 256 ? 256 : len + 1;
    char *nb = (char *)(realloc)(jit__raise_msg, cap);
    if (!nb) {
      len = jit__raise_msg_cap > 0 ? jit__raise_msg_cap - 1 : 0;
    } else {
      jit__raise_msg = nb;
      jit__raise_msg_cap = cap;
    }
  }
  if (jit__raise_msg_cap > 0) {
    memcpy(jit__raise_msg, bytes, len);
    jit__raise_msg[len] = '\0';
  }
}

/* Raise with an explicit message length (messages may hold NULs from
 * formatted text). `msg == NULL` leaves the wording to the interpreter. */
static void jit_raise_n(int kind, const char *id, const char *msg,
                        size_t msg_len) __attribute__((noreturn, cold));
static void jit_raise_n(int kind, const char *id, const char *msg,
                        size_t msg_len) {
  jit__raise_kind = kind;
  jit__raise_loc = jit__loc;
  size_t id_len = id ? strlen(id) : 0;
  if (id_len >= sizeof(jit__raise_id)) id_len = sizeof(jit__raise_id) - 1;
  if (id_len > 0) memcpy(jit__raise_id, id, id_len);
  jit__raise_id[id_len] = '\0';
  jit__raise_has_msg = msg != NULL;
  jit__raise_set_msg(msg ? msg : "", msg ? msg_len : 0);
  longjmp(jit__raise_buf, 1);
}

static void jit_raise(int kind, const char *id, const char *msg)
  __attribute__((noreturn, cold));
static void jit_raise(int kind, const char *id, const char *msg) {
  jit_raise_n(kind, id, msg, msg ? strlen(msg) : 0);
}

/* A JIT limitation: the interpreter re-runs the unit. `why` is a
 * diagnostic (surfaced through the host's bail warning). */
static void jit_raise_bail(const char *why) __attribute__((noreturn, cold));
static void jit_raise_bail(const char *why) {
  jit_raise(JIT_RAISE_BAIL, "", why);
}

/* `jit_raise_bail` with a printf-formatted reason. */
static void jit_raise_bailf(const char *fmt, ...)
  __attribute__((noreturn, cold, format(printf, 1, 2)));
static void jit_raise_bailf(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  jit_raise(JIT_RAISE_BAIL, "", buf);
}

/* ── Cancel ────────────────────────────────────────────────────────── */

static int32_t jit__cancel_none = 0;
static int32_t *jit__cancel = &jit__cancel_none;

/* Install the host's cancel flag (NULL: none). Exported. */
void jit_set_cancel(int32_t *flag) {
  jit__cancel = flag ? flag : &jit__cancel_none;
}

static void jit_raise_cancel(void) __attribute__((noreturn, cold));
static void jit_raise_cancel(void) {
  jit_raise(JIT_RAISE_CANCEL, "", NULL);
}

/* Raise a cancel if the host's flag is set. A relaxed atomic load: the
 * flag is written by another thread, and the load must not be hoisted out
 * of the loop that polls it. */
#define JIT_POLL_CANCEL()                                                   \
  do {                                                                      \
    if (__builtin_expect(__atomic_load_n(jit__cancel, __ATOMIC_RELAXED), 0)) \
      jit_raise_cancel();                                                   \
  } while (0)

/* A MATLAB error with the interpreter's exact message and identifier. */
static void jit_raise_error(const char *id, const char *msg)
  __attribute__((noreturn, cold));
static void jit_raise_error(const char *id, const char *msg) {
  jit_raise(JIT_RAISE_ERROR, id, msg);
}

/* `jit_raise_error` with a printf-formatted message. */
static void jit_raise_errorf(const char *id, const char *fmt, ...)
  __attribute__((noreturn, cold, format(printf, 2, 3)));
static void jit_raise_errorf(const char *id, const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  jit_raise(JIT_RAISE_ERROR, id, buf);
}

/* A MATLAB error whose wording the JIT does not reproduce: the
 * interpreter re-runs the unit to raise it. */
static void jit_raise_error_deferred(void) __attribute__((noreturn, cold));
static void jit_raise_error_deferred(void) {
  jit_raise(JIT_RAISE_ERROR, "", NULL);
}

/* An invalid dimension argument (`sum(x, 0)`). */
static void jit_raise_bad_dim(void) __attribute__((noreturn, cold));
static void jit_raise_bad_dim(void) {
  jit_raise_error("MATLAB:getdimarg:invalidDim",
                  "Dimension argument must be a positive integer scalar, a vector of unique positive integers, or 'all'.");
}

/* Operands whose sizes do not broadcast (the interpreter's
 * `sizeMismatchError`, ops/broadcast.ts). */
static void jit_raise_size_mismatch(void) __attribute__((noreturn, cold));
static void jit_raise_size_mismatch(void) {
  jit_raise_error("MATLAB:sizeDimensionsMustMatch",
                  "Arrays have incompatible sizes for this operation.");
}

/* ── Host accessors (exported) ─────────────────────────────────────── */

/* Clear the status and arm the location. Called by the host-entry wrapper
 * before it calls the spec. */
static void jit__raise_reset(void) {
  jit__raise_kind = JIT_RAISE_NONE;
  jit__loc = -1;
}

int jit_raise_kind(void) { return jit__raise_kind; }
const char *jit_raise_id(void) { return jit__raise_id; }
/* NULL when the wording is left to the interpreter. */
const char *jit_raise_msg(void) {
  return jit__raise_has_msg ? jit__raise_msg : NULL;
}
long long jit_raise_loc(void) { return jit__raise_loc; }
