/* mtoc2 runtime: numbl-compatible format engine shared by fprintf and
 * sprintf.
 *
 * Mirrors numbl's `sprintfFormat`
 * (numbl/src/numbl-core/helpers/string.ts) byte-for-byte:
 *   - spec set: d i u f e E g G x X o s c %
 *   - flags  - + 0 # space; precision `.N`; width digits; `*` consumes
 *     the next numeric arg as a width
 *   - escapes \n \t \\ interpreted AT FORMAT TIME (matching the numbl
 *     lexer, which preserves backslash bytes verbatim)
 *   - numeric tensors flatten column-major into the scalar stream;
 *     format string cycles through the args until the stream is
 *     exhausted (with a "no arg consumed this pass" guard against
 *     infinite output)
 *
 * Output goes through a caller-supplied `mtoc2__writer_fn` callback so
 * the same walker drives fprintf (FILE* sink) and sprintf (growable-
 * buffer sink).
 *
 * Argument transport: each call hands the walker an array of
 * `mtoc2_fprintf_arg_t` (a tagged union). The codegen builds the array
 * at the call site via a C99 compound literal, keeping the call shape
 * a single C expression even for variadic arities.
 *
 * v1 scope: real-only. Complex args are rejected at lowering, so the
 * engine carries no complex paths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Tag identifying the payload variant of a single mtoc2_fprintf_arg_t.
 * Values are stable — the engine switches on them. */
enum {
  MTOC2_FA_DOUBLE = 1,
  MTOC2_FA_TEXT = 3,    /* string (double-quoted): Number()-parsed if numeric */
  MTOC2_FA_TENSOR = 4,  /* real tensor */
  MTOC2_FA_CHAR = 5     /* char array (single-quoted): code point under %d/%c */
};

typedef struct {
  int kind;
  union {
    double d;
    mtoc2_text_view_t t;
    const mtoc2_tensor_t *tensor;
  } u;
} mtoc2_fprintf_arg_t;

/* Per-slot value handed to one format spec, after expanding any
 * tensor argument into its flattened element stream. */
typedef struct {
  int kind;           /* MTOC2_FA_DOUBLE / _TEXT */
  double d;
  mtoc2_text_view_t t;
} mtoc2__slot_t;

/* Iterator state over the (tagged) arg array with lazy column-major
 * tensor flattening. */
typedef struct {
  const mtoc2_fprintf_arg_t *args;
  int nargs;
  int arg_idx;             /* next un-touched entry of args[] */
  int in_tensor;           /* 1 iff currently iterating a flattened tensor */
  const mtoc2_tensor_t *cur_t;
  long t_idx;
  long t_count;
} mtoc2__arg_iter_t;

typedef void (*mtoc2__writer_fn)(void *ctx, const char *bytes, long len);

/* Advance the iterator to the next scalar slot. Returns 0 when no
 * more slots are available (caller stops cycling). */
static int mtoc2__next_slot(mtoc2__arg_iter_t *it, mtoc2__slot_t *out) {
  while (1) {
    if (it->in_tensor) {
      if (it->t_idx >= it->t_count) {
        it->in_tensor = 0;
        it->cur_t = (const mtoc2_tensor_t *)0;
        continue;
      }
      const mtoc2_tensor_t *t = it->cur_t;
      long i = it->t_idx++;
      out->kind = MTOC2_FA_DOUBLE;
      out->d = t->real[i];
      return 1;
    }
    if (it->arg_idx >= it->nargs) return 0;
    const mtoc2_fprintf_arg_t *a = &it->args[it->arg_idx++];
    switch (a->kind) {
      case MTOC2_FA_DOUBLE:
        out->kind = MTOC2_FA_DOUBLE;
        out->d = a->u.d;
        return 1;
      case MTOC2_FA_TEXT:
        out->kind = MTOC2_FA_TEXT;
        out->t = a->u.t;
        return 1;
      case MTOC2_FA_CHAR:
        out->kind = MTOC2_FA_CHAR;
        out->t = a->u.t;
        return 1;
      case MTOC2_FA_TENSOR: {
        const mtoc2_tensor_t *t = a->u.tensor;
        long n = 1;
        for (int k = 0; k < t->ndim; k++) n *= t->dims[k];
        if (n <= 0) continue;   /* empty tensor — skip */
        it->in_tensor = 1;
        it->cur_t = t;
        it->t_idx = 0;
        it->t_count = n;
        break;
      }
      default:
        return 0;
    }
  }
}

/* True iff the iterator has any remaining slot. */
static int mtoc2__iter_has_more(const mtoc2__arg_iter_t *it) {
  if (it->in_tensor && it->t_idx < it->t_count) return 1;
  if (it->arg_idx < it->nargs) return 1;
  return 0;
}

/* Number of slots consumed so far — used to detect a no-progress
 * pass over a format with no specs. */
static long mtoc2__iter_consumed(const mtoc2__arg_iter_t *it) {
  long n = (long)it->arg_idx;
  if (it->in_tensor) n = (long)(it->arg_idx - 1) + (long)it->t_idx;
  return n;
}

/* Emit a small byte buffer through the writer. */
static void mtoc2__emit_bytes(mtoc2__writer_fn writer, void *ctx,
                              const char *bytes, long len) {
  if (len > 0) writer(ctx, bytes, len);
}

/* Pad `s` (length `slen`) to `width` using `pad_char` (' ' or '0'),
 * respecting left-align. */
static void mtoc2__emit_padded(mtoc2__writer_fn writer, void *ctx,
                               const char *s, long slen,
                               long width, char pad_char, int left_align) {
  if (width <= slen) {
    mtoc2__emit_bytes(writer, ctx, s, slen);
    return;
  }
  long pad_len = width - slen;
  if (left_align) {
    mtoc2__emit_bytes(writer, ctx, s, slen);
    for (long i = 0; i < pad_len; i++) writer(ctx, " ", 1);
  } else {
    for (long i = 0; i < pad_len; i++) writer(ctx, &pad_char, 1);
    mtoc2__emit_bytes(writer, ctx, s, slen);
  }
}

/* Parse a numeric STRING the way JS `Number(s)` does — used when a
 * string-typed (double-quoted) arg meets a numeric/%c conversion. Trim
 * ASCII whitespace, then strtod the whole remainder: empty/whitespace
 * is 0 (Number("")===0), trailing non-numeric bytes give NaN
 * (Number("12abc")===NaN). Matches format_engine.js's `Number(v)`. */
static double mtoc2__string_to_number(const char *data, long len) {
  if (len <= 0 || !data) return 0.0;
  long s = 0, e = len;
  while (s < e && (data[s] == ' ' || data[s] == '\t' || data[s] == '\n' ||
                   data[s] == '\r' || data[s] == '\f' || data[s] == '\v'))
    s++;
  while (e > s && (data[e - 1] == ' ' || data[e - 1] == '\t' ||
                   data[e - 1] == '\n' || data[e - 1] == '\r' ||
                   data[e - 1] == '\f' || data[e - 1] == '\v'))
    e--;
  if (s == e) return 0.0;
  long n = e - s;
  char buf[64];
  char *tmp = buf;
  if (n + 1 > (long)sizeof(buf)) {
    tmp = (char *)malloc((size_t)n + 1);
    if (!tmp) return (double)NAN;
  }
  memcpy(tmp, data + s, (size_t)n);
  tmp[n] = '\0';
  char *endp = tmp;
  double v = strtod(tmp, &endp);
  double result = (*endp == '\0') ? v : (double)NAN;
  if (tmp != buf) free(tmp);
  return result;
}

/* Convert a format slot to the numeric value a numeric/%c conversion
 * expects, mirroring format_engine.js's `toNumber`: a char yields its
 * first code point (the single-char case; the interpreter/JS have no
 * scalar form for a multi-char char), a string parses via Number(), a
 * double is itself. A bare-0.0 fallback here would have made every
 * char/string slot print 0 / a NUL byte. */
static double mtoc2__slot_to_double(const mtoc2__slot_t *slot) {
  if (slot->kind == MTOC2_FA_CHAR)
    return slot->t.len >= 1 ? (double)(unsigned char)slot->t.data[0] : 0.0;
  if (slot->kind == MTOC2_FA_TEXT)
    return mtoc2__string_to_number(slot->t.data, slot->t.len);
  return slot->d;
}

/* Parse the spec body (between '%' and the type char) — fills
 * `width` (0 if absent), `prec` (-1 if absent), and the flag set.
 * Resolves '*' for width by consuming one numeric slot via the
 * iterator. Returns 1 on success, 0 if the iterator ran out while
 * resolving '*'. */
static int mtoc2__parse_spec(const char *spec, long spec_len,
                             mtoc2__arg_iter_t *it,
                             long *out_width, int *out_prec,
                             int *out_has_plus, int *out_has_space,
                             int *out_left_align, int *out_zero_pad) {
  long width = 0;
  int prec = -1;
  int has_plus = 0, has_space = 0, left_align = 0, zero_pad = 0;
  int in_prec = 0;
  /* spec begins with '%'; iterate over its body. */
  for (long i = 1; i < spec_len; i++) {
    char c = spec[i];
    if (in_prec) {
      if (c >= '0' && c <= '9') {
        if (prec < 0) prec = 0;
        prec = prec * 10 + (c - '0');
        continue;
      }
      in_prec = 0;
    }
    if (c == '.') {
      in_prec = 1;
      prec = 0;
      continue;
    }
    if (c == '-') { left_align = 1; continue; }
    if (c == '+') { has_plus = 1; continue; }
    if (c == ' ') { has_space = 1; continue; }
    if (c == '#') { /* parsed but no behavior */ continue; }
    if (c == '0' && width == 0) { zero_pad = 1; continue; }
    if (c >= '0' && c <= '9') {
      width = width * 10 + (c - '0');
      continue;
    }
    if (c == '*') {
      mtoc2__slot_t slot;
      if (!mtoc2__next_slot(it, &slot)) return 0;
      double v = mtoc2__slot_to_double(&slot);
      long n = (long)floor(v + 0.5);
      width = n > 0 ? n : 0;
      continue;
    }
    /* Anything else is unexpected — ignore for width. */
  }
  if (zero_pad && left_align) zero_pad = 0;
  *out_width = width;
  *out_prec = prec;
  *out_has_plus = has_plus;
  *out_has_space = has_space;
  *out_left_align = left_align;
  *out_zero_pad = zero_pad;
  return 1;
}

/* Format a real number as JS `String(n)` would — used by %s on a
 * numeric arg (numbl's `toString` path). */
static long mtoc2__num_to_str(char *buf, size_t cap, double x) {
  if (isnan(x)) return (long)snprintf(buf, cap, "NaN");
  if (isinf(x)) return (long)snprintf(buf, cap, x > 0 ? "Infinity" : "-Infinity");
  if (fabs(x) < 1e16 && x == (double)(long long)x) {
    return (long)snprintf(buf, cap, "%lld", (long long)x);
  }
  long n = (long)snprintf(buf, cap, "%.17g", x);
  return n;
}

/* Emit one `\` escape sequence — numbl interprets only \n, \t, \\
 * at format time; any other `\x` passes through verbatim. */
static long mtoc2__emit_escape(mtoc2__writer_fn writer, void *ctx,
                                char next) {
  switch (next) {
    case 'n': writer(ctx, "\n", 1); return 1;
    case 't': writer(ctx, "\t", 1); return 1;
    case '\\': writer(ctx, "\\", 1); return 1;
    default: {
      char buf[2] = { '\\', next };
      writer(ctx, buf, 2);
      return 1;
    }
  }
}

/* Emit a %d / %i / %u slot. Mirrors numbl's int branch:
 *   - non-integer (or negative for %u) → fall back to %e prec=6
 *     with 2-digit-padded exponent.
 *   - otherwise render |x| as decimal, prepend sign per flags,
 *     and apply width/zero-pad. */
static void mtoc2__emit_int(mtoc2__writer_fn writer, void *ctx,
                             const char *spec, long spec_len,
                             mtoc2__arg_iter_t *it, char type_ch,
                             double raw) {
  int is_int = isfinite(raw) && raw == floor(raw) && fabs(raw) < 1e21;
  int can_int = is_int;
  if (type_ch == 'u' && raw < 0.0) can_int = 0;
  long width;
  int prec, has_plus, has_space, left_align, zero_pad;
  if (!mtoc2__parse_spec(spec, spec_len, it,
                         &width, &prec, &has_plus, &has_space,
                         &left_align, &zero_pad)) {
    return;
  }
  /* Non-finite under %d/%i/%u: numbl's int branch routes through
     raw.toExponential(6), and (Inf).toExponential() === "Infinity",
     (NaN) === "NaN" — not libc's "inf"/"-nan". Match the JS strings. */
  if (!isfinite(raw)) {
    const char *s = isnan(raw) ? "NaN" : (raw > 0 ? "Infinity" : "-Infinity");
    mtoc2__emit_padded(writer, ctx, s, (long)strlen(s), width,
                       ' ', left_align);
    return;
  }
  if (!can_int) {
    char ebuf[64];
    long elen = (long)snprintf(ebuf, sizeof(ebuf), "%.*e",
                                prec >= 0 ? prec : 6, raw);
    /* Pad exponent to ≥2 digits (numbl rule). */
    if (elen >= 3) {
      char *e = strchr(ebuf, 'e');
      if (e && (e[1] == '+' || e[1] == '-') &&
          e[2] >= '0' && e[2] <= '9' && e[3] == '\0') {
        char tail = e[2];
        e[2] = '0';
        e[3] = tail;
        e[4] = '\0';
        elen++;
      }
    }
    mtoc2__emit_padded(writer, ctx, ebuf, elen, width,
                       zero_pad ? '0' : ' ', left_align);
    return;
  }
  /* Integer formatting. |raw| < ~9e18 fits a long long; larger
     integer-valued doubles (up to the 1e21 cap above) overflow it, so
     print the full decimal of the magnitude with %.0f — matching JS
     String(Math.abs(n)) (e.g. 1e19 -> "10000000000000000000"). */
  char ibuf[400];
  long ilen;
  int is_neg = raw < 0.0;
  if (fabs(raw) < 9.0e18) {
    long long n_int = (long long)raw;
    if (n_int < 0) n_int = -n_int;
    ilen = (long)snprintf(ibuf, sizeof(ibuf), "%lld", n_int);
  } else {
    ilen = (long)snprintf(ibuf, sizeof(ibuf), "%.0f", fabs(raw));
  }
  char sign_ch = 0;
  if (is_neg) sign_ch = '-';
  else if (has_plus) sign_ch = '+';
  else if (has_space) sign_ch = ' ';
  long sign_len = sign_ch ? 1 : 0;
  long total = sign_len + ilen;
  if (width <= total) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, ibuf, ilen);
    return;
  }
  long pad = width - total;
  if (left_align) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, ibuf, ilen);
    for (long i = 0; i < pad; i++) writer(ctx, " ", 1);
  } else if (zero_pad) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    for (long i = 0; i < pad; i++) writer(ctx, "0", 1);
    mtoc2__emit_bytes(writer, ctx, ibuf, ilen);
  } else {
    for (long i = 0; i < pad; i++) writer(ctx, " ", 1);
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, ibuf, ilen);
  }
}

/* Emit %f / %e / %E / %g / %G slot — defers to snprintf for the
 * numeric body, applies sign-prefix and width afterwards. */
static void mtoc2__emit_float(mtoc2__writer_fn writer, void *ctx,
                               const char *spec, long spec_len,
                               mtoc2__arg_iter_t *it, char type_ch,
                               double x) {
  long width;
  int prec, has_plus, has_space, left_align, zero_pad;
  if (!mtoc2__parse_spec(spec, spec_len, it,
                         &width, &prec, &has_plus, &has_space,
                         &left_align, &zero_pad)) {
    return;
  }
  if (!isfinite(x)) {
    /* numbl's %f/%e/%g non-finite spelling comes from numStr(): "Inf" /
       "-Inf" / "NaN" — NOT libc's "Infinity"/"inf". (The %d path differs:
       see mtoc2__emit_int, which uses toExponential -> "Infinity".) */
    const char *s = isnan(x) ? "NaN" : (x > 0 ? "Inf" : "-Inf");
    long slen = (long)strlen(s);
    mtoc2__emit_padded(writer, ctx, s, slen, width, ' ', left_align);
    return;
  }
  int default_prec = 6;
  if (prec < 0) prec = default_prec;

  char buf[96];
  long len = 0;
  if (type_ch == 'f') {
    len = (long)snprintf(buf, sizeof(buf), "%.*f", prec, fabs(x));
  } else if (type_ch == 'e' || type_ch == 'E') {
    len = (long)snprintf(buf, sizeof(buf), "%.*e", prec, fabs(x));
    char *e = strchr(buf, 'e');
    if (e && (e[1] == '+' || e[1] == '-') &&
        e[2] >= '0' && e[2] <= '9' && e[3] == '\0') {
      char tail = e[2];
      e[2] = '0';
      e[3] = tail;
      e[4] = '\0';
      len++;
    }
    if (type_ch == 'E') {
      for (long i = 0; i < len; i++) {
        if (buf[i] == 'e') buf[i] = 'E';
      }
    }
  } else {
    /* %g / %G */
    int gprec = prec > 0 ? prec : 1;
    if (x == 0.0) {
      buf[0] = '0';
      buf[1] = '\0';
      len = 1;
    } else {
      double ax = fabs(x);
      int exp10 = (int)floor(log10(ax));
      if (exp10 < -4 || exp10 >= gprec) {
        len = (long)snprintf(buf, sizeof(buf), "%.*e", gprec - 1, ax);
        char *e = strchr(buf, 'e');
        if (e) {
          char *m_end = e - 1;
          while (m_end > buf && *m_end == '0') m_end--;
          if (m_end > buf && *m_end == '.') m_end--;
          long m_keep = m_end - buf + 1;
          long e_len = (long)strlen(e);
          memmove(buf + m_keep, e, (size_t)e_len + 1);
          len = m_keep + e_len;
          e = buf + m_keep;
          if ((e[1] == '+' || e[1] == '-') &&
              e[2] >= '0' && e[2] <= '9' && e[3] == '\0') {
            char tail = e[2];
            e[2] = '0';
            e[3] = tail;
            e[4] = '\0';
            len++;
          }
        }
      } else {
        int frac = gprec - 1 - exp10;
        if (frac < 0) frac = 0;
        len = (long)snprintf(buf, sizeof(buf), "%.*f", frac, ax);
        if (strchr(buf, '.')) {
          while (len > 0 && buf[len - 1] == '0') len--;
          if (len > 0 && buf[len - 1] == '.') len--;
          buf[len] = '\0';
        }
      }
    }
    if (type_ch == 'G') {
      for (long i = 0; i < len; i++) {
        if (buf[i] >= 'a' && buf[i] <= 'z') buf[i] = (char)(buf[i] - 32);
      }
    }
  }
  char sign_ch = 0;
  if (x < 0.0) sign_ch = '-';
  else if (has_plus) sign_ch = '+';
  else if (has_space) sign_ch = ' ';
  long sign_len = sign_ch ? 1 : 0;
  long total = sign_len + len;
  if (width <= total) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, buf, len);
    return;
  }
  long pad = width - total;
  if (left_align) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, buf, len);
    for (long i = 0; i < pad; i++) writer(ctx, " ", 1);
  } else if (zero_pad) {
    if (sign_ch) writer(ctx, &sign_ch, 1);
    for (long i = 0; i < pad; i++) writer(ctx, "0", 1);
    mtoc2__emit_bytes(writer, ctx, buf, len);
  } else {
    for (long i = 0; i < pad; i++) writer(ctx, " ", 1);
    if (sign_ch) writer(ctx, &sign_ch, 1);
    mtoc2__emit_bytes(writer, ctx, buf, len);
  }
}

/* Emit %x / %X / %o — round-to-int absolute value, then convert. */
static void mtoc2__emit_xo(mtoc2__writer_fn writer, void *ctx,
                           const char *spec, long spec_len,
                           mtoc2__arg_iter_t *it, char type_ch,
                           double raw) {
  long width;
  int prec, has_plus, has_space, left_align, zero_pad;
  if (!mtoc2__parse_spec(spec, spec_len, it,
                         &width, &prec, &has_plus, &has_space,
                         &left_align, &zero_pad)) {
    return;
  }
  /* Round the SIGNED value with Math.round semantics (ties toward +Inf),
     then take the magnitude — matching JS Math.abs(Math.round(raw)).
     Rounding fabs(raw) instead flips negative half-way values (e.g.
     -1.5 -> 2 instead of the correct 1). */
  long long sn = (long long)floor(raw + 0.5);
  long long n = sn < 0 ? -sn : sn;
  char buf[32];
  long len = 0;
  if (type_ch == 'o') {
    len = (long)snprintf(buf, sizeof(buf), "%llo", n);
  } else {
    len = (long)snprintf(buf, sizeof(buf), "%llx", n);
    if (type_ch == 'X') {
      for (long i = 0; i < len; i++) {
        if (buf[i] >= 'a' && buf[i] <= 'f') buf[i] = (char)(buf[i] - 32);
      }
    }
  }
  mtoc2__emit_padded(writer, ctx, buf, len, width,
                     zero_pad ? '0' : ' ', left_align);
}

/* Emit %s — text view goes through verbatim; a numeric slot routes
 * through `String(n)` semantics. */
static void mtoc2__emit_s(mtoc2__writer_fn writer, void *ctx,
                          const char *spec, long spec_len,
                          mtoc2__arg_iter_t *it, const mtoc2__slot_t *slot) {
  long width;
  int prec, has_plus, has_space, left_align, zero_pad;
  if (!mtoc2__parse_spec(spec, spec_len, it,
                         &width, &prec, &has_plus, &has_space,
                         &left_align, &zero_pad)) {
    return;
  }
  const char *data = (const char *)0;
  long slen = 0;
  char numbuf[64];
  if (slot->kind == MTOC2_FA_TEXT || slot->kind == MTOC2_FA_CHAR) {
    data = slot->t.data;
    slen = slot->t.len > 0 ? slot->t.len : 0;
  } else {
    slen = mtoc2__num_to_str(numbuf, sizeof(numbuf), slot->d);
    data = numbuf;
  }
  mtoc2__emit_padded(writer, ctx, data, slen, width, ' ', left_align);
}

/* The main walker. Loops over `fmt` consuming args; cycles through
 * the format string until the arg stream is exhausted, with a
 * "made no progress this pass" guard against infinite output on a
 * format with no `%` specifiers. */
static void mtoc2__format_walk(mtoc2__writer_fn writer, void *ctx,
                               mtoc2_text_view_t fmt,
                               int nargs, const mtoc2_fprintf_arg_t *args) {
  mtoc2__arg_iter_t it;
  it.args = args;
  it.nargs = nargs;
  it.arg_idx = 0;
  it.in_tensor = 0;
  it.cur_t = (const mtoc2_tensor_t *)0;
  it.t_idx = 0;
  it.t_count = 0;

  long fmt_len = fmt.len > 0 ? fmt.len : 0;
  if (fmt_len == 0 || !fmt.data) return;

  do {
    long start_consumed = mtoc2__iter_consumed(&it);
    long i = 0;
    int out_of_args = 0;
    while (i < fmt_len && !out_of_args) {
      char c = fmt.data[i];
      if (c == '%' && i + 1 < fmt_len) {
        long spec_start = i;
        i++;
        while (i < fmt_len &&
               !strchr("dfigeEsoxXuc%", fmt.data[i])) {
          i++;
        }
        if (i >= fmt_len) break;
        char type_ch = fmt.data[i++];
        long spec_len = i - 1 - spec_start;
        const char *spec = fmt.data + spec_start;
        if (type_ch == '%') {
          writer(ctx, "%", 1);
          continue;
        }
        mtoc2__slot_t slot;
        if (!mtoc2__next_slot(&it, &slot)) {
          out_of_args = 1;
          break;
        }
        double dval = mtoc2__slot_to_double(&slot);
        switch (type_ch) {
          case 'd':
          case 'i':
          case 'u':
            mtoc2__emit_int(writer, ctx, spec, spec_len, &it,
                            type_ch, dval);
            break;
          case 'f':
          case 'e':
          case 'E':
          case 'g':
          case 'G':
            mtoc2__emit_float(writer, ctx, spec, spec_len, &it,
                              type_ch, dval);
            break;
          case 'x':
          case 'X':
          case 'o':
            mtoc2__emit_xo(writer, ctx, spec, spec_len, &it,
                           type_ch, dval);
            break;
          case 's':
            mtoc2__emit_s(writer, ctx, spec, spec_len, &it, &slot);
            break;
          case 'c': {
            int code = (int)floor(dval + 0.5);
            if (code < 0) code = 0;
            if (code > 0x10FFFF) code = 0;
            if (code < 0x80) {
              char ch = (char)code;
              writer(ctx, &ch, 1);
            } else if (code < 0x800) {
              char b[2];
              b[0] = (char)(0xC0 | (code >> 6));
              b[1] = (char)(0x80 | (code & 0x3F));
              writer(ctx, b, 2);
            } else if (code < 0x10000) {
              char b[3];
              b[0] = (char)(0xE0 | (code >> 12));
              b[1] = (char)(0x80 | ((code >> 6) & 0x3F));
              b[2] = (char)(0x80 | (code & 0x3F));
              writer(ctx, b, 3);
            } else {
              char b[4];
              b[0] = (char)(0xF0 | (code >> 18));
              b[1] = (char)(0x80 | ((code >> 12) & 0x3F));
              b[2] = (char)(0x80 | ((code >> 6) & 0x3F));
              b[3] = (char)(0x80 | (code & 0x3F));
              writer(ctx, b, 4);
            }
            break;
          }
          default:
            mtoc2__emit_bytes(writer, ctx, spec, spec_len + 1);
            break;
        }
      } else if (c == '\\' && i + 1 < fmt_len) {
        i++;
        i += mtoc2__emit_escape(writer, ctx, fmt.data[i]);
      } else {
        writer(ctx, fmt.data + i, 1);
        i++;
      }
    }
    /* No progress this pass — stop to avoid infinite output. */
    if (mtoc2__iter_consumed(&it) == start_consumed) break;
  } while (mtoc2__iter_has_more(&it));
}
