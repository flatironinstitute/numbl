/*
 * numbl_plot.h: plot arrays from a C or C++ program in numbl's browser
 * viewer, using MATLAB plot commands. Single-header library: get it with
 *
 *   numbl c-header > numbl_plot.h
 *
 * and in exactly ONE .c or .cpp file of your program, before any other
 * #include, write
 *
 *   #define NUMBL_PLOT_IMPLEMENTATION
 *   #include "numbl_plot.h"
 *
 * Other files include it without the #define. Compile with -pthread (and
 * -lm). POSIX only (Linux, macOS).
 *
 *   numbl_engine *e = numbl_open();
 *   numbl_eval_sync(e, "colormap jet");        // setup: waits, never dropped
 *   for (...) {
 *     numbl_put_matrix(e, "u", u, rows, cols); // column-major, copied
 *     numbl_put_scalar(e, "t", t);
 *     numbl_eval(e, "imagesc(u); title(sprintf('t = %g', t))"); // returns
 *   }
 *   numbl_close(e);
 *
 * Frames are sent by a background thread, so numbl_eval costs the caller
 * only the copies made by numbl_put_*. Variables persist in the server's
 * workspace between calls, as at the MATLAB prompt. Call the functions of
 * one engine from one thread at a time.
 *
 * Each frame is one HTTP/1.1 POST to the server's /eval, over a keep-alive
 * connection, carrying a binary frame (src/plotFrame.ts in numbl):
 *   "NBF1" | uint32 header length | JSON header {code, vars} | payload
 * A queued frame from numbl_eval is replaced by a newer numbl_eval frame
 * with the same code, which inherits the variables only the older frame had.
 * Frames from numbl_eval_sync are never replaced.
 */

/* The implementation needs POSIX declarations, which must be requested
 * before the first system header. */
#if defined(NUMBL_PLOT_IMPLEMENTATION) && !defined(_POSIX_C_SOURCE) && \
    !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(NUMBL_PLOT_IMPLEMENTATION) && defined(__APPLE__) && \
    !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE /* for SO_NOSIGPIPE */
#endif

#ifndef NUMBL_PLOT_H
#define NUMBL_PLOT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct numbl_engine numbl_engine;

/* Connect to the plot server at $NUMBL_PLOT_URL, or, when it is unset, start
 * one by running `$NUMBL_CMD plot-server` (NUMBL_CMD defaults to "numbl").
 * Returns NULL on failure, with a message on stderr. */
numbl_engine *numbl_open(void);

/* Queue a rows x cols double matrix stored column-major (Fortran / MATLAB
 * order: element (i, j) at data[i + j * rows]). The data is copied, so the
 * caller may change it right away. It is sent with the next numbl_eval or
 * numbl_eval_sync. A C array declared as a[rows][cols] is row-major: pass it
 * as a cols x rows matrix and transpose in MATLAB (u.'). Returns 0 on
 * success. */
int numbl_put_matrix(numbl_engine *e, const char *name, const double *data,
                     size_t rows, size_t cols);

/* Same, from single-precision data (converted to double by the server). */
int numbl_put_matrix_f32(numbl_engine *e, const char *name, const float *data,
                         size_t rows, size_t cols);

/* Queue a scalar. Returns 0 on success. */
int numbl_put_scalar(numbl_engine *e, const char *name, double value);

/* Queue the variables put so far plus `code` as one frame and return at
 * once. If an earlier numbl_eval frame with the same code is still waiting
 * to be sent, this frame replaces it (keeping the variables only the earlier
 * frame had), so a fast loop sends only the latest data and never waits.
 * Displayed text and errors are printed by the sender thread (stdout and
 * stderr). Returns 0 if queued. */
int numbl_eval(numbl_engine *e, const char *code);

/* Wait until all earlier frames are sent, then run `code` with the queued
 * variables and wait for the reply. Never replaced. Returns 0 on success,
 * -1 on error (see numbl_last_error). */
int numbl_eval_sync(numbl_engine *e, const char *code);

/* Message for the most recent error, or "" if none. */
const char *numbl_last_error(numbl_engine *e);

/* Frames sent so far (including ones that failed), and numbl_eval frames
 * replaced by newer ones before they were sent. */
void numbl_get_stats(numbl_engine *e, size_t *sent, size_t *replaced);

/* Send the frames still queued, disconnect and free the engine. If
 * numbl_open started the server, keep it running so the figures stay
 * visible, and wait for it to exit (Ctrl+C). */
void numbl_close(numbl_engine *e);

#ifdef __cplusplus
}
#endif

#endif

/* ═══ Implementation ═══════════════════════════════════════════════════ */

#if defined(NUMBL_PLOT_IMPLEMENTATION) && !defined(NUMBL_PLOT_IMPLEMENTED)
#define NUMBL_PLOT_IMPLEMENTED

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0 /* macOS: SO_NOSIGPIPE is set on the socket instead */
#endif

#define NBP_ERR_SIZE 1024

/* ── growable byte buffer ─────────────────────────────────────────────── */

typedef struct {
  char *p;
  size_t len, cap;
} nbp_buf_t;

static int nbp_buf_reserve(nbp_buf_t *b, size_t extra) {
  if (b->len + extra <= b->cap) return 0;
  size_t cap = b->cap ? b->cap : 256;
  while (cap < b->len + extra) cap *= 2;
  char *p = (char *)realloc(b->p, cap);
  if (!p) return -1;
  b->p = p;
  b->cap = cap;
  return 0;
}

static int nbp_buf_append(nbp_buf_t *b, const void *data, size_t n) {
  if (n == 0) return 0;
  if (nbp_buf_reserve(b, n)) return -1;
  memcpy(b->p + b->len, data, n);
  b->len += n;
  return 0;
}

static int nbp_buf_printf(nbp_buf_t *b, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0 || nbp_buf_reserve(b, (size_t)n + 1)) return -1;
  va_start(ap, fmt);
  vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap);
  va_end(ap);
  b->len += (size_t)n;
  return 0;
}

static int nbp_buf_json_string(nbp_buf_t *b, const char *s) {
  if (nbp_buf_append(b, "\"", 1)) return -1;
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    int rc;
    if (c == '"') rc = nbp_buf_append(b, "\\\"", 2);
    else if (c == '\\') rc = nbp_buf_append(b, "\\\\", 2);
    else if (c < 0x20) rc = nbp_buf_printf(b, "\\u%04x", c);
    else rc = nbp_buf_append(b, s, 1);
    if (rc) return -1;
  }
  return nbp_buf_append(b, "\"", 1);
}

static void nbp_buf_free(nbp_buf_t *b) {
  free(b->p);
  b->p = NULL;
  b->len = b->cap = 0;
}

/* ── frames ───────────────────────────────────────────────────────────── */

typedef struct {
  char *name;
  size_t rows, cols, offset, bytes;
  const char *dtype;
} nbp_var_t;

typedef struct nbp_frame {
  char *code;
  nbp_var_t *vars;
  size_t nvars, varcap;
  nbp_buf_t payload;
  int sync;            /* from numbl_eval_sync: never replaced */
  int done, rc;        /* sync result, set by the sender */
  char err[NBP_ERR_SIZE];  /* sync error message */
  struct nbp_frame *next;
} nbp_frame_t;

static nbp_frame_t *nbp_frame_new(void) { return (nbp_frame_t *)calloc(1, sizeof(nbp_frame_t)); }

static void nbp_frame_free(nbp_frame_t *f) {
  if (!f) return;
  for (size_t i = 0; i < f->nvars; i++) free(f->vars[i].name);
  free(f->vars);
  free(f->code);
  nbp_buf_free(&f->payload);
  free(f);
}

static int nbp_frame_has_var(const nbp_frame_t *f, const char *name) {
  for (size_t i = 0; i < f->nvars; i++)
    if (strcmp(f->vars[i].name, name) == 0) return 1;
  return 0;
}

static int nbp_frame_add_var(nbp_frame_t *f, const char *name, const void *data,
                         size_t rows, size_t cols, size_t bytes,
                         const char *dtype) {
  if (f->nvars == f->varcap) {
    size_t cap = f->varcap ? 2 * f->varcap : 8;
    nbp_var_t *v = (nbp_var_t *)realloc(f->vars, cap * sizeof *v);
    if (!v) return -1;
    f->vars = v;
    f->varcap = cap;
  }
  nbp_var_t *v = &f->vars[f->nvars];
  v->name = (char *)malloc(strlen(name) + 1);
  if (!v->name) return -1;
  strcpy(v->name, name);
  v->rows = rows;
  v->cols = cols;
  v->bytes = bytes;
  v->dtype = dtype;
  v->offset = f->payload.len;
  if (nbp_buf_append(&f->payload, data, bytes)) {
    free(v->name);
    return -1;
  }
  f->nvars++;
  return 0;
}

/* ── engine ───────────────────────────────────────────────────────────── */

struct numbl_engine {
  char host[256];
  char port[16];
  pid_t child;        /* > 0 when numbl_open started the server */
  FILE *child_out;    /* the started server's stdout */

  /* caller thread only */
  nbp_frame_t *pending;   /* variables queued by numbl_put_* */
  char err_copy[NBP_ERR_SIZE];

  /* shared, guarded by mu */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  nbp_frame_t *head, *tail; /* frames waiting to be sent */
  int closing;
  size_t n_sent, n_replaced;
  char last_err[NBP_ERR_SIZE];
  char printed_err[NBP_ERR_SIZE]; /* last error printed, to skip repeats */

  /* sender thread only (and numbl_open before it starts) */
  pthread_t thread;
  int thread_started;
  int fd;             /* -1 when not connected */
  nbp_buf_t header, req, resp;
  char text[65536];
  char serr[NBP_ERR_SIZE];
};

static void nbp_set_serr(numbl_engine *e, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(e->serr, sizeof e->serr, fmt, ap);
  va_end(ap);
}

/* Record and print an error (deduplicating consecutive repeats). Call with
 * mu held. */
static void nbp_report_locked(numbl_engine *e, const char *msg) {
  snprintf(e->last_err, sizeof e->last_err, "%s", msg);
  if (strcmp(msg, e->printed_err) != 0) {
    fprintf(stderr, "numbl: %s\n", msg);
    snprintf(e->printed_err, sizeof e->printed_err, "%s", msg);
  }
}

static void nbp_report(numbl_engine *e, const char *fmt, ...) {
  char msg[NBP_ERR_SIZE];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  pthread_mutex_lock(&e->mu);
  nbp_report_locked(e, msg);
  pthread_mutex_unlock(&e->mu);
}

/* Parse "http://host:port/..." into e->host and e->port. */
static int nbp_parse_url(numbl_engine *e, const char *url) {
  const char *p = url;
  if (strncmp(p, "http://", 7) != 0) {
    nbp_set_serr(e, "unsupported plot server URL '%s' (expected http://)", url);
    return -1;
  }
  p += 7;
  size_t h = strcspn(p, ":/");
  if (h == 0 || h >= sizeof e->host) {
    nbp_set_serr(e, "bad host in plot server URL '%s'", url);
    return -1;
  }
  memcpy(e->host, p, h);
  e->host[h] = '\0';
  p += h;
  if (*p == ':') {
    p++;
    size_t n = strcspn(p, "/");
    if (n == 0 || n >= sizeof e->port) {
      nbp_set_serr(e, "bad port in plot server URL '%s'", url);
      return -1;
    }
    memcpy(e->port, p, n);
    e->port[n] = '\0';
  } else {
    strcpy(e->port, "80");
  }
  return 0;
}

static int nbp_connect_server(numbl_engine *e) {
  struct addrinfo hints, *res, *ai;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  int rc = getaddrinfo(e->host, e->port, &hints, &res);
  if (rc != 0) {
    nbp_set_serr(e, "cannot resolve %s: %s", e->host, gai_strerror(rc));
    return -1;
  }
  int fd = -1;
  for (ai = res; ai; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) {
    nbp_set_serr(e, "cannot connect to plot server at %s:%s: %s", e->host,
             e->port, strerror(errno));
    return -1;
  }
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  e->fd = fd;
  return 0;
}

static void nbp_disconnect(numbl_engine *e) {
  if (e->fd >= 0) close(e->fd);
  e->fd = -1;
}

/* Start `$NUMBL_CMD plot-server` and read the URL it prints on stdout. */
static int nbp_spawn_server(numbl_engine *e) {
  const char *cmd = getenv("NUMBL_CMD");
  if (!cmd || !*cmd) cmd = "numbl";
  nbp_buf_t sh = {NULL, 0, 0};
  if (nbp_buf_printf(&sh, "exec %s plot-server", cmd)) {
    nbp_set_serr(e, "out of memory");
    return -1;
  }
  int pfd[2];
  if (pipe(pfd) != 0) {
    nbp_set_serr(e, "pipe: %s", strerror(errno));
    nbp_buf_free(&sh);
    return -1;
  }
  fflush(NULL);
  pid_t pid = fork();
  if (pid < 0) {
    nbp_set_serr(e, "fork: %s", strerror(errno));
    close(pfd[0]);
    close(pfd[1]);
    nbp_buf_free(&sh);
    return -1;
  }
  if (pid == 0) {
    dup2(pfd[1], STDOUT_FILENO);
    close(pfd[0]);
    close(pfd[1]);
    execl("/bin/sh", "sh", "-c", sh.p, (char *)NULL);
    _exit(127);
  }
  nbp_buf_free(&sh);
  close(pfd[1]);
  FILE *f = fdopen(pfd[0], "r");
  if (!f) {
    nbp_set_serr(e, "fdopen: %s", strerror(errno));
    close(pfd[0]);
    return -1;
  }
  char line[512];
  const char *key = "NUMBL_PLOT_URL=";
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, key, strlen(key)) != 0) continue;
    line[strcspn(line, "\r\n")] = '\0';
    e->child = pid;
    e->child_out = f; /* kept open until the server exits */
    return nbp_parse_url(e, line + strlen(key));
  }
  fclose(f);
  waitpid(pid, NULL, 0);
  nbp_set_serr(e,
           "'%s plot-server' exited without printing its URL "
           "(set NUMBL_CMD to a numbl that has the plot-server command)",
           cmd);
  return -1;
}

/* ── HTTP (sender thread) ─────────────────────────────────────────────── */

static int nbp_send_all(int fd, const char *p, size_t n) {
  while (n > 0) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

/* Decode the JSON string value of "key" in `json` into `out` (truncated to
 * outsize). Returns 0 if found. Assumes compact JSON (no spaces). */
static int nbp_json_get_string(const char *json, const char *key, char *out,
                           size_t outsize) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\":\"", key);
  const char *p = strstr(json, pat);
  if (!p) return -1;
  p += strlen(pat);
  size_t n = 0;
  while (*p && *p != '"') {
    char c = *p++;
    if (c == '\\' && *p) {
      char esc = *p++;
      switch (esc) {
      case 'n': c = '\n'; break;
      case 't': c = '\t'; break;
      case 'r': c = '\r'; break;
      case 'b': c = '\b'; break;
      case 'f': c = '\f'; break;
      case 'u': {
        unsigned cp = 0;
        for (int i = 0; i < 4 && *p; i++, p++) {
          char h = *p;
          cp = cp * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
        }
        /* UTF-8 encode (BMP only; surrogates become '?') */
        unsigned char u[3];
        size_t k = 0;
        if (cp < 0x80) u[k++] = (unsigned char)cp;
        else if (cp < 0x800) {
          u[k++] = (unsigned char)(0xC0 | (cp >> 6));
          u[k++] = (unsigned char)(0x80 | (cp & 0x3F));
        } else if (cp >= 0xD800 && cp < 0xE000) u[k++] = '?';
        else {
          u[k++] = (unsigned char)(0xE0 | (cp >> 12));
          u[k++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
          u[k++] = (unsigned char)(0x80 | (cp & 0x3F));
        }
        for (size_t i = 0; i < k; i++)
          if (n + 1 < outsize) out[n++] = (char)u[i];
        continue;
      }
      default: c = esc; break; /* \" \\ \/ */
      }
    }
    if (n + 1 < outsize) out[n++] = c;
  }
  out[n] = '\0';
  return 0;
}

/* Read one HTTP response into e->resp (NUL-terminated). Sets *body to the
 * start of the body. Returns 0 on success, 1 if the connection closed before
 * any byte arrived (stale keep-alive, worth retrying), -1 on other errors. */
static int nbp_read_response(numbl_engine *e, int *status, char **body) {
  nbp_buf_t *r = &e->resp;
  r->len = 0;
  size_t head_end = 0, content_len = 0;
  int have_head = 0;
  for (;;) {
    if (have_head && r->len >= head_end + content_len) break;
    if (nbp_buf_reserve(r, 65536)) {
      nbp_set_serr(e, "out of memory");
      return -1;
    }
    ssize_t k = recv(e->fd, r->p + r->len, r->cap - r->len - 1, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) {
      if (r->len == 0) return 1;
      nbp_set_serr(e, "connection to plot server lost");
      return -1;
    }
    r->len += (size_t)k;
    r->p[r->len] = '\0';
    if (!have_head) {
      char *end = strstr(r->p, "\r\n\r\n");
      if (!end) continue;
      head_end = (size_t)(end - r->p) + 4;
      if (sscanf(r->p, "HTTP/%*s %d", status) != 1) {
        nbp_set_serr(e, "bad HTTP response from plot server");
        return -1;
      }
      int found = 0;
      for (char *line = strstr(r->p, "\r\n"); line && line < end;
           line = strstr(line + 2, "\r\n")) {
        if (strncasecmp(line + 2, "Content-Length:", 15) == 0) {
          content_len = strtoul(line + 2 + 15, NULL, 10);
          found = 1;
        }
      }
      if (!found) {
        nbp_set_serr(e, "plot server reply has no Content-Length");
        return -1;
      }
      have_head = 1;
    }
  }
  r->p[head_end + content_len] = '\0';
  *body = r->p + head_end;
  return 0;
}

/* One request/response round. Returns as nbp_read_response. */
static int nbp_round_trip(numbl_engine *e, const nbp_frame_t *f, int *status,
                      char **body) {
  if (e->fd < 0 && nbp_connect_server(e)) return -1;
  if (nbp_send_all(e->fd, e->req.p, e->req.len) ||
      nbp_send_all(e->fd, f->payload.p, f->payload.len)) {
    return 1; /* likely a keep-alive connection the server closed */
  }
  return nbp_read_response(e, status, body);
}

/* Send one frame and handle the reply. Returns 0 on success; on failure, -1
 * with the message in e->serr. */
static int nbp_send_frame(numbl_engine *e, const nbp_frame_t *f) {
  e->serr[0] = '\0';
  nbp_buf_t *h = &e->header;
  h->len = 0;
  int bad = nbp_buf_append(h, "{\"code\":", 8) || nbp_buf_json_string(h, f->code) ||
            nbp_buf_append(h, ",\"vars\":{", 9);
  for (size_t i = 0; i < f->nvars && !bad; i++) {
    const nbp_var_t *v = &f->vars[i];
    bad = (i > 0 && nbp_buf_append(h, ",", 1)) || nbp_buf_json_string(h, v->name) ||
          nbp_buf_printf(h,
                     ":{\"shape\":[%zu,%zu],\"dtype\":\"%s\",\"offset\":%zu}",
                     v->rows, v->cols, v->dtype, v->offset);
  }
  bad = bad || nbp_buf_append(h, "}}", 2);
  if (bad || h->len > UINT32_MAX) {
    nbp_set_serr(e, "cannot build request");
    return -1;
  }

  nbp_buf_t *q = &e->req;
  q->len = 0;
  uint32_t hlen = (uint32_t)h->len;
  unsigned char le[4] = {(unsigned char)hlen, (unsigned char)(hlen >> 8),
                         (unsigned char)(hlen >> 16),
                         (unsigned char)(hlen >> 24)};
  if (nbp_buf_printf(q,
                 "POST /eval HTTP/1.1\r\nHost: %s:%s\r\n"
                 "Content-Type: application/octet-stream\r\n"
                 "Content-Length: %zu\r\n\r\n",
                 e->host, e->port, 8 + h->len + f->payload.len) ||
      nbp_buf_append(q, "NBF1", 4) || nbp_buf_append(q, le, 4) ||
      nbp_buf_append(q, h->p, h->len)) {
    nbp_set_serr(e, "out of memory");
    return -1;
  }

  int status = 0;
  char *body = NULL;
  int r = nbp_round_trip(e, f, &status, &body);
  if (r == 1) { /* reconnect once */
    nbp_disconnect(e);
    r = nbp_round_trip(e, f, &status, &body);
  }
  if (r != 0) {
    if (r == 1) nbp_set_serr(e, "plot server closed the connection");
    nbp_disconnect(e);
    return -1;
  }

  if (nbp_json_get_string(body, "output", e->text, sizeof e->text) == 0 &&
      e->text[0]) {
    fputs(e->text, stdout);
    fflush(stdout);
  }
  if (status == 200 && strstr(body, "\"ok\":true")) return 0;
  if (nbp_json_get_string(body, "error", e->text, sizeof e->text) == 0)
    nbp_set_serr(e, "%s", e->text);
  else
    nbp_set_serr(e, "plot server returned HTTP %d", status);
  return -1;
}

static void *nbp_sender_main(void *arg) {
  numbl_engine *e = (numbl_engine *)arg;
  pthread_mutex_lock(&e->mu);
  for (;;) {
    while (!e->head && !e->closing) pthread_cond_wait(&e->cv, &e->mu);
    nbp_frame_t *f = e->head;
    if (!f) break; /* closing, queue empty */
    e->head = f->next;
    if (!e->head) e->tail = NULL;
    pthread_mutex_unlock(&e->mu);

    int rc = nbp_send_frame(e, f);

    pthread_mutex_lock(&e->mu);
    e->n_sent++;
    if (rc == 0) {
      e->printed_err[0] = '\0';
    } else {
      nbp_report_locked(e, e->serr);
    }
    if (f->sync) {
      f->rc = rc;
      snprintf(f->err, sizeof f->err, "%s", rc ? e->serr : "");
      f->done = 1;
      pthread_cond_broadcast(&e->cv);
    } else {
      nbp_frame_free(f);
    }
  }
  pthread_mutex_unlock(&e->mu);
  return NULL;
}

/* ── public API ───────────────────────────────────────────────────────── */

numbl_engine *numbl_open(void) {
  const uint16_t probe = 1;
  if (*(const unsigned char *)&probe != 1) {
    fprintf(stderr, "numbl: big-endian hosts are not supported\n");
    return NULL;
  }
  numbl_engine *e = (numbl_engine *)calloc(1, sizeof *e);
  if (!e) return NULL;
  e->fd = -1;
  pthread_mutex_init(&e->mu, NULL);
  pthread_cond_init(&e->cv, NULL);
  const char *url = getenv("NUMBL_PLOT_URL");
  int rc = (url && *url) ? nbp_parse_url(e, url) : nbp_spawn_server(e);
  if (rc == 0) rc = nbp_connect_server(e);
  if (rc == 0 && pthread_create(&e->thread, NULL, nbp_sender_main, e) != 0) {
    nbp_set_serr(e, "cannot start sender thread");
    rc = -1;
  }
  if (rc != 0) {
    fprintf(stderr, "numbl: %s\n", e->serr);
    if (e->child > 0) { /* started a server we cannot use */
      kill(e->child, SIGTERM);
      waitpid(e->child, NULL, 0);
      e->child = 0;
    }
    numbl_close(e);
    return NULL;
  }
  e->thread_started = 1;
  return e;
}

static int nbp_put(numbl_engine *e, const char *name, const void *data,
               size_t rows, size_t cols, size_t itemsize, const char *dtype) {
  if (!e || !name || (!data && rows * cols > 0)) return -1;
  if (cols != 0 && rows > SIZE_MAX / cols / itemsize) {
    nbp_report(e, "'%s': matrix too large", name);
    return -1;
  }
  if (!e->pending && !(e->pending = nbp_frame_new())) goto oom;
  if (nbp_frame_add_var(e->pending, name, data, rows, cols, rows * cols * itemsize,
                    dtype))
    goto oom;
  return 0;
oom:
  nbp_report(e, "out of memory");
  return -1;
}

int numbl_put_matrix(numbl_engine *e, const char *name, const double *data,
                     size_t rows, size_t cols) {
  return nbp_put(e, name, data, rows, cols, sizeof(double), "float64");
}

int numbl_put_matrix_f32(numbl_engine *e, const char *name, const float *data,
                         size_t rows, size_t cols) {
  return nbp_put(e, name, data, rows, cols, sizeof(float), "float32");
}

int numbl_put_scalar(numbl_engine *e, const char *name, double value) {
  return nbp_put(e, name, &value, 1, 1, sizeof(double), "float64");
}

/* Detach the queued variables as a frame running `code`. */
static nbp_frame_t *nbp_take_frame(numbl_engine *e, const char *code) {
  nbp_frame_t *f = e->pending ? e->pending : nbp_frame_new();
  e->pending = NULL;
  if (f) f->code = (char *)malloc(strlen(code) + 1);
  if (!f || !f->code) {
    nbp_frame_free(f);
    nbp_report(e, "out of memory");
    return NULL;
  }
  strcpy(f->code, code);
  return f;
}

int numbl_eval(numbl_engine *e, const char *code) {
  if (!e || !code) return -1;
  nbp_frame_t *f = nbp_take_frame(e, code);
  if (!f) return -1;
  int rc = 0;
  pthread_mutex_lock(&e->mu);
  nbp_frame_t *prev = NULL, *q = e->head;
  while (q && (q->sync || strcmp(q->code, f->code) != 0)) {
    prev = q;
    q = q->next;
  }
  if (q) {
    /* Replace q by f in place; f inherits the variables only q had. */
    for (size_t i = 0; i < q->nvars && rc == 0; i++) {
      const nbp_var_t *v = &q->vars[i];
      if (!nbp_frame_has_var(f, v->name))
        rc = nbp_frame_add_var(f, v->name, q->payload.p + v->offset, v->rows,
                           v->cols, v->bytes, v->dtype);
    }
    if (rc == 0) {
      f->next = q->next;
      if (prev) prev->next = f;
      else e->head = f;
      if (e->tail == q) e->tail = f;
      e->n_replaced++;
    } else {
      nbp_report_locked(e, "out of memory");
      q = NULL; /* keep q queued */
    }
  }
  if (rc == 0) {
    if (!q) {
      f->next = NULL;
      if (e->tail) e->tail->next = f;
      else e->head = f;
      e->tail = f;
    }
    pthread_cond_broadcast(&e->cv);
  }
  pthread_mutex_unlock(&e->mu);
  if (rc != 0) nbp_frame_free(f);
  else nbp_frame_free(q);
  return rc;
}

int numbl_eval_sync(numbl_engine *e, const char *code) {
  if (!e || !code) return -1;
  nbp_frame_t *f = nbp_take_frame(e, code);
  if (!f) return -1;
  f->sync = 1;
  pthread_mutex_lock(&e->mu);
  if (e->tail) e->tail->next = f;
  else e->head = f;
  e->tail = f;
  pthread_cond_broadcast(&e->cv);
  while (!f->done) pthread_cond_wait(&e->cv, &e->mu);
  int rc = f->rc;
  if (rc) snprintf(e->last_err, sizeof e->last_err, "%s", f->err);
  pthread_mutex_unlock(&e->mu);
  nbp_frame_free(f);
  return rc;
}

const char *numbl_last_error(numbl_engine *e) {
  if (!e) return "no engine";
  pthread_mutex_lock(&e->mu);
  snprintf(e->err_copy, sizeof e->err_copy, "%s", e->last_err);
  pthread_mutex_unlock(&e->mu);
  return e->err_copy;
}

void numbl_get_stats(numbl_engine *e, size_t *sent, size_t *replaced) {
  if (!e) return;
  pthread_mutex_lock(&e->mu);
  if (sent) *sent = e->n_sent;
  if (replaced) *replaced = e->n_replaced;
  pthread_mutex_unlock(&e->mu);
}

void numbl_close(numbl_engine *e) {
  if (!e) return;
  if (e->thread_started) {
    pthread_mutex_lock(&e->mu);
    e->closing = 1;
    pthread_cond_broadcast(&e->cv);
    pthread_mutex_unlock(&e->mu);
    pthread_join(e->thread, NULL); /* sends what is still queued */
  }
  nbp_disconnect(e);
  nbp_frame_free(e->pending);
  while (e->head) {
    nbp_frame_t *n = e->head->next;
    nbp_frame_free(e->head);
    e->head = n;
  }
  nbp_buf_free(&e->header);
  nbp_buf_free(&e->req);
  nbp_buf_free(&e->resp);
  if (e->child > 0) {
    fprintf(stderr,
            "numbl: figures remain at http://%s:%s/ ; press Ctrl+C to exit\n",
            e->host, e->port);
    waitpid(e->child, NULL, 0);
  }
  if (e->child_out) fclose(e->child_out);
  pthread_cond_destroy(&e->cv);
  pthread_mutex_destroy(&e->mu);
  free(e);
}

#endif /* NUMBL_PLOT_IMPLEMENTATION */
