# numbl_plot: live plots from C and C++

A single-header library that lets a C or C++ program (Fortran later) plot its arrays while it runs, using MATLAB plot commands, in numbl's browser viewer. It follows the MATLAB Engine API: put named arrays, then evaluate MATLAB code.

User guide: https://numbl.org/docs/live-plots-from-c

## Getting started

With numbl installed (`npm install -g numbl`), get the header into your project:

```bash
numbl c-header > numbl_plot.h
```

In exactly one source file, define `NUMBL_PLOT_IMPLEMENTATION` before including it (and before any other `#include`); other files include it plainly:

```c
#define NUMBL_PLOT_IMPLEMENTATION
#include "numbl_plot.h"

int main(void) {
  numbl_engine *e = numbl_open();
  numbl_eval_sync(e, "colormap hot");        /* setup: waits for the reply */
  for (int step = 0; step < nsteps; step++) {
    advance(u);
    if (step % 50 == 0) {
      numbl_put_matrix(e, "u", u, ny, nx);   /* column-major, copied */
      numbl_put_scalar(e, "t", t);
      numbl_eval(e, "imagesc(u); colorbar; title(sprintf('t = %g', t))");  /* returns at once */
    }
  }
  numbl_close(e);
}
```

Build with no extra paths or libraries beyond threads:

```bash
cc sim.c -pthread -lm -o sim
```

The header compiles as C99 or later and as C++. Re-run `numbl c-header` after upgrading numbl so the client matches the server.

The code runs in a numbl process (`numbl plot-server`) that keeps one workspace, so variables persist between calls and the snippet can use any numbl function (`log10(abs(u))`, `u(1:4:end, :)`, `subplot`, ...).

## Running

`numbl_open` starts `numbl plot-server`, which opens the viewer in the browser. When the program finishes, `numbl_close` keeps the server running so the figures stay visible; press Ctrl+C to exit.

To keep one viewer across runs, start the server yourself and point programs at it:

```bash
numbl plot-server --plot-port 8790
NUMBL_PLOT_URL=http://127.0.0.1:8790/ ./sim
```

The demo in this directory (a 2-D heat equation with a moving source) builds with `make` and runs as `./examples/heat2d`.

## Viewer controls

While the program runs, 2-D plots in the viewer can be explored without stopping it; the view stays as set while new frames arrive.

- Scroll: zoom about the cursor (Shift: x only, Ctrl: y only). Drag: pan.
- Scroll or drag on the colorbar: change the color limits.
- Double-click or "Reset view": back to the automatic limits.
- Keys, after clicking the plot: arrows pan, `+` `-` zoom, `[` `]` contrast, `{` `}` shift the colors, `r` reset.

## Environment

- `NUMBL_PLOT_URL`: connect to this server instead of starting one.
- `NUMBL_CMD`: command used to start the server (default `numbl`), e.g. `npx tsx /path/to/numbl/src/cli.ts` for a source checkout.

## API

See the comments in `numbl_plot.h`. In short:

- `numbl_put_matrix(e, name, data, rows, cols)`: double data, column-major (element `(i, j)` at `data[i + j*rows]`). A row-major C array `a[rows][cols]` can be passed as `cols x rows` and transposed in MATLAB with `u.'`.
- `numbl_put_matrix_f32`: same from float data (half the bytes on the wire).
- `numbl_put_scalar(e, name, value)`.
- `numbl_eval(e, code)`: queues the variables put so far plus `code` as one frame and returns at once. A background thread sends frames in order. If a frame with the same code is still waiting when a new one arrives, the new one replaces it and inherits any variables only the older frame had (such as a grid put once). So a loop that produces frames faster than the viewer can draw sends only the latest data, and different snippets (two figures, say) each keep their own latest frame.
- `numbl_eval_sync(e, code)`: waits for earlier frames, runs `code`, and returns 0, or -1 with the message in `numbl_last_error(e)`. Never replaced; use it for setup, or to check that a snippet works.
- `numbl_get_stats(e, &sent, &replaced)`: frame counts.
- Text the code displays goes to stdout; errors go to stderr, printed once while the same error repeats.
- Call the functions of one engine from one thread at a time.

## Notes

- The cost on the calling thread is copying the arrays in `numbl_put_*`. In the heat demo, 0.2% of run time for a 150 x 200 grid plotted every 50 steps, and 6.6% for a 1000 x 1000 grid plotted every 2 steps.
- The server replies to a frame only after the browser has taken the plot data, so the frame rate adapts to what the browser can draw (about 28 frames/s for 150 x 200, 9 for 1000 x 1000 on a laptop).
- The server listens on 127.0.0.1 only, because `/eval` runs arbitrary code. For a simulation on a remote machine, forward the port with ssh (`ssh -L 8790:127.0.0.1:8790 host`).
- Wire format: one HTTP POST to `/eval` per frame, with a binary frame described in `src/plotFrame.ts`.
- POSIX only (Linux, macOS); little-endian hosts.
