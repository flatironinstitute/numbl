/*
 * numbl_plot: plot arrays from a C/C++/Fortran program in numbl's browser
 * viewer, using MATLAB plot commands.
 *
 *   numbl_engine *e = numbl_open();
 *   numbl_eval_sync(e, "colormap hot");        // setup: waits, never dropped
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
 * one engine from one thread at a time. See README.md.
 */
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
