/*
 * numbl_plot: C client for `numbl plot-server`. See numbl_plot.h.
 *
 * Each numbl_eval / numbl_eval_sync becomes one frame (src/plotFrame.ts in
 * the numbl repo), sent as an HTTP/1.1 POST to /eval over a keep-alive
 * connection:
 *
 *   "NBF1" | uint32 header length | JSON header {code, vars} | payload
 *
 * Every variable, scalars included, travels as a raw array in the payload.
 *
 * Threads: the caller builds frames (numbl_put_* copies the data) and queues
 * them; a sender thread sends them one at a time, in order. A queued frame
 * from numbl_eval is replaced by a newer numbl_eval frame with the same code,
 * which inherits the variables only the older frame had. Frames from
 * numbl_eval_sync are never replaced.
 */
#define _POSIX_C_SOURCE 200809L

#include "numbl_plot.h"

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

#define ERR_SIZE 1024

/* ── growable byte buffer ─────────────────────────────────────────────── */

typedef struct {
  char *p;
  size_t len, cap;
} buf_t;

static int buf_reserve(buf_t *b, size_t extra) {
  if (b->len + extra <= b->cap) return 0;
  size_t cap = b->cap ? b->cap : 256;
  while (cap < b->len + extra) cap *= 2;
  char *p = realloc(b->p, cap);
  if (!p) return -1;
  b->p = p;
  b->cap = cap;
  return 0;
}

static int buf_append(buf_t *b, const void *data, size_t n) {
  if (n == 0) return 0;
  if (buf_reserve(b, n)) return -1;
  memcpy(b->p + b->len, data, n);
  b->len += n;
  return 0;
}

static int buf_printf(buf_t *b, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0 || buf_reserve(b, (size_t)n + 1)) return -1;
  va_start(ap, fmt);
  vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap);
  va_end(ap);
  b->len += (size_t)n;
  return 0;
}

static int buf_json_string(buf_t *b, const char *s) {
  if (buf_append(b, "\"", 1)) return -1;
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    int rc;
    if (c == '"') rc = buf_append(b, "\\\"", 2);
    else if (c == '\\') rc = buf_append(b, "\\\\", 2);
    else if (c < 0x20) rc = buf_printf(b, "\\u%04x", c);
    else rc = buf_append(b, s, 1);
    if (rc) return -1;
  }
  return buf_append(b, "\"", 1);
}

static void buf_free(buf_t *b) {
  free(b->p);
  b->p = NULL;
  b->len = b->cap = 0;
}

/* ── frames ───────────────────────────────────────────────────────────── */

typedef struct {
  char *name;
  size_t rows, cols, offset, bytes;
  const char *dtype;
} var_t;

typedef struct frame {
  char *code;
  var_t *vars;
  size_t nvars, varcap;
  buf_t payload;
  int sync;            /* from numbl_eval_sync: never replaced */
  int done, rc;        /* sync result, set by the sender */
  char err[ERR_SIZE];  /* sync error message */
  struct frame *next;
} frame_t;

static frame_t *frame_new(void) { return calloc(1, sizeof(frame_t)); }

static void frame_free(frame_t *f) {
  if (!f) return;
  for (size_t i = 0; i < f->nvars; i++) free(f->vars[i].name);
  free(f->vars);
  free(f->code);
  buf_free(&f->payload);
  free(f);
}

static int frame_has_var(const frame_t *f, const char *name) {
  for (size_t i = 0; i < f->nvars; i++)
    if (strcmp(f->vars[i].name, name) == 0) return 1;
  return 0;
}

static int frame_add_var(frame_t *f, const char *name, const void *data,
                         size_t rows, size_t cols, size_t bytes,
                         const char *dtype) {
  if (f->nvars == f->varcap) {
    size_t cap = f->varcap ? 2 * f->varcap : 8;
    var_t *v = realloc(f->vars, cap * sizeof *v);
    if (!v) return -1;
    f->vars = v;
    f->varcap = cap;
  }
  var_t *v = &f->vars[f->nvars];
  v->name = malloc(strlen(name) + 1);
  if (!v->name) return -1;
  strcpy(v->name, name);
  v->rows = rows;
  v->cols = cols;
  v->bytes = bytes;
  v->dtype = dtype;
  v->offset = f->payload.len;
  if (buf_append(&f->payload, data, bytes)) {
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
  frame_t *pending;   /* variables queued by numbl_put_* */
  char err_copy[ERR_SIZE];

  /* shared, guarded by mu */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  frame_t *head, *tail; /* frames waiting to be sent */
  int closing;
  size_t n_sent, n_replaced;
  char last_err[ERR_SIZE];
  char printed_err[ERR_SIZE]; /* last error printed, to skip repeats */

  /* sender thread only (and numbl_open before it starts) */
  pthread_t thread;
  int thread_started;
  int fd;             /* -1 when not connected */
  buf_t header, req, resp;
  char text[65536];
  char serr[ERR_SIZE];
};

static void set_serr(numbl_engine *e, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(e->serr, sizeof e->serr, fmt, ap);
  va_end(ap);
}

/* Record and print an error (deduplicating consecutive repeats). Call with
 * mu held. */
static void report_locked(numbl_engine *e, const char *msg) {
  snprintf(e->last_err, sizeof e->last_err, "%s", msg);
  if (strcmp(msg, e->printed_err) != 0) {
    fprintf(stderr, "numbl: %s\n", msg);
    snprintf(e->printed_err, sizeof e->printed_err, "%s", msg);
  }
}

static void report(numbl_engine *e, const char *fmt, ...) {
  char msg[ERR_SIZE];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  pthread_mutex_lock(&e->mu);
  report_locked(e, msg);
  pthread_mutex_unlock(&e->mu);
}

/* Parse "http://host:port/..." into e->host and e->port. */
static int parse_url(numbl_engine *e, const char *url) {
  const char *p = url;
  if (strncmp(p, "http://", 7) != 0) {
    set_serr(e, "unsupported plot server URL '%s' (expected http://)", url);
    return -1;
  }
  p += 7;
  size_t h = strcspn(p, ":/");
  if (h == 0 || h >= sizeof e->host) {
    set_serr(e, "bad host in plot server URL '%s'", url);
    return -1;
  }
  memcpy(e->host, p, h);
  e->host[h] = '\0';
  p += h;
  if (*p == ':') {
    p++;
    size_t n = strcspn(p, "/");
    if (n == 0 || n >= sizeof e->port) {
      set_serr(e, "bad port in plot server URL '%s'", url);
      return -1;
    }
    memcpy(e->port, p, n);
    e->port[n] = '\0';
  } else {
    strcpy(e->port, "80");
  }
  return 0;
}

static int connect_server(numbl_engine *e) {
  struct addrinfo hints, *res, *ai;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  int rc = getaddrinfo(e->host, e->port, &hints, &res);
  if (rc != 0) {
    set_serr(e, "cannot resolve %s: %s", e->host, gai_strerror(rc));
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
    set_serr(e, "cannot connect to plot server at %s:%s: %s", e->host,
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

static void disconnect(numbl_engine *e) {
  if (e->fd >= 0) close(e->fd);
  e->fd = -1;
}

/* Start `$NUMBL_CMD plot-server` and read the URL it prints on stdout. */
static int spawn_server(numbl_engine *e) {
  const char *cmd = getenv("NUMBL_CMD");
  if (!cmd || !*cmd) cmd = "numbl";
  buf_t sh = {0};
  if (buf_printf(&sh, "exec %s plot-server", cmd)) {
    set_serr(e, "out of memory");
    return -1;
  }
  int pfd[2];
  if (pipe(pfd) != 0) {
    set_serr(e, "pipe: %s", strerror(errno));
    buf_free(&sh);
    return -1;
  }
  fflush(NULL);
  pid_t pid = fork();
  if (pid < 0) {
    set_serr(e, "fork: %s", strerror(errno));
    close(pfd[0]);
    close(pfd[1]);
    buf_free(&sh);
    return -1;
  }
  if (pid == 0) {
    dup2(pfd[1], STDOUT_FILENO);
    close(pfd[0]);
    close(pfd[1]);
    execl("/bin/sh", "sh", "-c", sh.p, (char *)NULL);
    _exit(127);
  }
  buf_free(&sh);
  close(pfd[1]);
  FILE *f = fdopen(pfd[0], "r");
  if (!f) {
    set_serr(e, "fdopen: %s", strerror(errno));
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
    return parse_url(e, line + strlen(key));
  }
  fclose(f);
  waitpid(pid, NULL, 0);
  set_serr(e,
           "'%s plot-server' exited without printing its URL "
           "(set NUMBL_CMD to a numbl that has the plot-server command)",
           cmd);
  return -1;
}

/* ── HTTP (sender thread) ─────────────────────────────────────────────── */

static int send_all(int fd, const char *p, size_t n) {
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
static int json_get_string(const char *json, const char *key, char *out,
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
static int read_response(numbl_engine *e, int *status, char **body) {
  buf_t *r = &e->resp;
  r->len = 0;
  size_t head_end = 0, content_len = 0;
  int have_head = 0;
  for (;;) {
    if (have_head && r->len >= head_end + content_len) break;
    if (buf_reserve(r, 65536)) {
      set_serr(e, "out of memory");
      return -1;
    }
    ssize_t k = recv(e->fd, r->p + r->len, r->cap - r->len - 1, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) {
      if (r->len == 0) return 1;
      set_serr(e, "connection to plot server lost");
      return -1;
    }
    r->len += (size_t)k;
    r->p[r->len] = '\0';
    if (!have_head) {
      char *end = strstr(r->p, "\r\n\r\n");
      if (!end) continue;
      head_end = (size_t)(end - r->p) + 4;
      if (sscanf(r->p, "HTTP/%*s %d", status) != 1) {
        set_serr(e, "bad HTTP response from plot server");
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
        set_serr(e, "plot server reply has no Content-Length");
        return -1;
      }
      have_head = 1;
    }
  }
  r->p[head_end + content_len] = '\0';
  *body = r->p + head_end;
  return 0;
}

/* One request/response round. Returns as read_response. */
static int round_trip(numbl_engine *e, const frame_t *f, int *status,
                      char **body) {
  if (e->fd < 0 && connect_server(e)) return -1;
  if (send_all(e->fd, e->req.p, e->req.len) ||
      send_all(e->fd, f->payload.p, f->payload.len)) {
    return 1; /* likely a keep-alive connection the server closed */
  }
  return read_response(e, status, body);
}

/* Send one frame and handle the reply. Returns 0 on success; on failure, -1
 * with the message in e->serr. */
static int send_frame(numbl_engine *e, const frame_t *f) {
  e->serr[0] = '\0';
  buf_t *h = &e->header;
  h->len = 0;
  int bad = buf_append(h, "{\"code\":", 8) || buf_json_string(h, f->code) ||
            buf_append(h, ",\"vars\":{", 9);
  for (size_t i = 0; i < f->nvars && !bad; i++) {
    const var_t *v = &f->vars[i];
    bad = (i > 0 && buf_append(h, ",", 1)) || buf_json_string(h, v->name) ||
          buf_printf(h,
                     ":{\"shape\":[%zu,%zu],\"dtype\":\"%s\",\"offset\":%zu}",
                     v->rows, v->cols, v->dtype, v->offset);
  }
  bad = bad || buf_append(h, "}}", 2);
  if (bad || h->len > UINT32_MAX) {
    set_serr(e, "cannot build request");
    return -1;
  }

  buf_t *q = &e->req;
  q->len = 0;
  uint32_t hlen = (uint32_t)h->len;
  unsigned char le[4] = {(unsigned char)hlen, (unsigned char)(hlen >> 8),
                         (unsigned char)(hlen >> 16),
                         (unsigned char)(hlen >> 24)};
  if (buf_printf(q,
                 "POST /eval HTTP/1.1\r\nHost: %s:%s\r\n"
                 "Content-Type: application/octet-stream\r\n"
                 "Content-Length: %zu\r\n\r\n",
                 e->host, e->port, 8 + h->len + f->payload.len) ||
      buf_append(q, "NBF1", 4) || buf_append(q, le, 4) ||
      buf_append(q, h->p, h->len)) {
    set_serr(e, "out of memory");
    return -1;
  }

  int status = 0;
  char *body = NULL;
  int r = round_trip(e, f, &status, &body);
  if (r == 1) { /* reconnect once */
    disconnect(e);
    r = round_trip(e, f, &status, &body);
  }
  if (r != 0) {
    if (r == 1) set_serr(e, "plot server closed the connection");
    disconnect(e);
    return -1;
  }

  if (json_get_string(body, "output", e->text, sizeof e->text) == 0 &&
      e->text[0]) {
    fputs(e->text, stdout);
    fflush(stdout);
  }
  if (status == 200 && strstr(body, "\"ok\":true")) return 0;
  if (json_get_string(body, "error", e->text, sizeof e->text) == 0)
    set_serr(e, "%s", e->text);
  else
    set_serr(e, "plot server returned HTTP %d", status);
  return -1;
}

static void *sender_main(void *arg) {
  numbl_engine *e = arg;
  pthread_mutex_lock(&e->mu);
  for (;;) {
    while (!e->head && !e->closing) pthread_cond_wait(&e->cv, &e->mu);
    frame_t *f = e->head;
    if (!f) break; /* closing, queue empty */
    e->head = f->next;
    if (!e->head) e->tail = NULL;
    pthread_mutex_unlock(&e->mu);

    int rc = send_frame(e, f);

    pthread_mutex_lock(&e->mu);
    e->n_sent++;
    if (rc == 0) {
      e->printed_err[0] = '\0';
    } else {
      report_locked(e, e->serr);
    }
    if (f->sync) {
      f->rc = rc;
      snprintf(f->err, sizeof f->err, "%s", rc ? e->serr : "");
      f->done = 1;
      pthread_cond_broadcast(&e->cv);
    } else {
      frame_free(f);
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
  numbl_engine *e = calloc(1, sizeof *e);
  if (!e) return NULL;
  e->fd = -1;
  pthread_mutex_init(&e->mu, NULL);
  pthread_cond_init(&e->cv, NULL);
  const char *url = getenv("NUMBL_PLOT_URL");
  int rc = (url && *url) ? parse_url(e, url) : spawn_server(e);
  if (rc == 0) rc = connect_server(e);
  if (rc == 0 && pthread_create(&e->thread, NULL, sender_main, e) != 0) {
    set_serr(e, "cannot start sender thread");
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

static int put(numbl_engine *e, const char *name, const void *data,
               size_t rows, size_t cols, size_t itemsize, const char *dtype) {
  if (!e || !name || (!data && rows * cols > 0)) return -1;
  if (cols != 0 && rows > SIZE_MAX / cols / itemsize) {
    report(e, "'%s': matrix too large", name);
    return -1;
  }
  if (!e->pending && !(e->pending = frame_new())) goto oom;
  if (frame_add_var(e->pending, name, data, rows, cols, rows * cols * itemsize,
                    dtype))
    goto oom;
  return 0;
oom:
  report(e, "out of memory");
  return -1;
}

int numbl_put_matrix(numbl_engine *e, const char *name, const double *data,
                     size_t rows, size_t cols) {
  return put(e, name, data, rows, cols, sizeof(double), "float64");
}

int numbl_put_matrix_f32(numbl_engine *e, const char *name, const float *data,
                         size_t rows, size_t cols) {
  return put(e, name, data, rows, cols, sizeof(float), "float32");
}

int numbl_put_scalar(numbl_engine *e, const char *name, double value) {
  return put(e, name, &value, 1, 1, sizeof(double), "float64");
}

/* Detach the queued variables as a frame running `code`. */
static frame_t *take_frame(numbl_engine *e, const char *code) {
  frame_t *f = e->pending ? e->pending : frame_new();
  e->pending = NULL;
  if (f) f->code = malloc(strlen(code) + 1);
  if (!f || !f->code) {
    frame_free(f);
    report(e, "out of memory");
    return NULL;
  }
  strcpy(f->code, code);
  return f;
}

int numbl_eval(numbl_engine *e, const char *code) {
  if (!e || !code) return -1;
  frame_t *f = take_frame(e, code);
  if (!f) return -1;
  int rc = 0;
  pthread_mutex_lock(&e->mu);
  frame_t *prev = NULL, *q = e->head;
  while (q && (q->sync || strcmp(q->code, f->code) != 0)) {
    prev = q;
    q = q->next;
  }
  if (q) {
    /* Replace q by f in place; f inherits the variables only q had. */
    for (size_t i = 0; i < q->nvars && rc == 0; i++) {
      const var_t *v = &q->vars[i];
      if (!frame_has_var(f, v->name))
        rc = frame_add_var(f, v->name, q->payload.p + v->offset, v->rows,
                           v->cols, v->bytes, v->dtype);
    }
    if (rc == 0) {
      f->next = q->next;
      if (prev) prev->next = f;
      else e->head = f;
      if (e->tail == q) e->tail = f;
      e->n_replaced++;
    } else {
      report_locked(e, "out of memory");
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
  if (rc != 0) frame_free(f);
  else frame_free(q);
  return rc;
}

int numbl_eval_sync(numbl_engine *e, const char *code) {
  if (!e || !code) return -1;
  frame_t *f = take_frame(e, code);
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
  frame_free(f);
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
  disconnect(e);
  frame_free(e->pending);
  while (e->head) {
    frame_t *n = e->head->next;
    frame_free(e->head);
    e->head = n;
  }
  buf_free(&e->header);
  buf_free(&e->req);
  buf_free(&e->resp);
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
