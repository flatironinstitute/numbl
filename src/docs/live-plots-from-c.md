# Live Plots from C and C++

A C or C++ simulation can plot its arrays while it runs, using MATLAB plot commands, in numbl's browser viewer. The simulation sends its arrays and a short MATLAB snippet such as `imagesc(u); colorbar`; numbl runs the snippet and updates the plot. While the simulation runs, the plot can be zoomed, panned and given new color limits.

The C interface follows the MATLAB Engine API: put named arrays, then evaluate MATLAB code. It is a single header file with no dependencies.

## Requirements

- [Node.js](https://nodejs.org) 18 or later, and numbl 0.4.23 or later: `npm install -g numbl`
- Linux or macOS (so far tested on Linux), with a C99 or C++ compiler

## Try the demo

The demo solves a 2-D heat equation with a moving heat source and plots the field and its middle row:

```bash
mkdir numbl-demo && cd numbl-demo
numbl c-header > numbl_plot.h
curl -O https://raw.githubusercontent.com/flatironinstitute/numbl/main/clients/c/examples/heat2d.c
cc heat2d.c -pthread -lm -o heat2d
./heat2d
```

A browser tab opens and updates as the simulation runs. When the program finishes, the figures stay open until you press Ctrl+C in the terminal.

## Viewer controls

The view you set stays in place while new frames arrive.

- Scroll: zoom about the cursor (Shift: x only, Ctrl: y only)
- Drag: pan
- Scroll or drag on the colorbar: change the color limits
- Double-click, or the "Reset view" button: back to the automatic limits
- Keys, after clicking the plot: arrows pan, `+` `-` zoom, `[` `]` contrast, `{` `}` shift the colors, `r` reset

## Use it in your program

Get the header with `numbl c-header > numbl_plot.h`. In exactly one source file, define `NUMBL_PLOT_IMPLEMENTATION` before including it, and before any other `#include`. Other files include it without the define.

```c
#define NUMBL_PLOT_IMPLEMENTATION
#include "numbl_plot.h"

int main(void) {
  numbl_engine *e = numbl_open();
  numbl_eval_sync(e, "colormap jet");
  for (int step = 0; step < nsteps; step++) {
    advance(u);
    if (step % 50 == 0) {
      numbl_put_matrix(e, "u", u, ny, nx);
      numbl_put_scalar(e, "t", step * dt);
      numbl_eval(e, "imagesc(u); colorbar; title(sprintf('t = %g', t))");
    }
  }
  numbl_close(e);
}
```

Compile with `-pthread` (and `-lm` if you use the math library); no other paths or libraries are needed.

The snippet is ordinary MATLAB run by numbl, so it can use `subplot`, `hold`, `log10(abs(u))`, `u(1:4:end, :)` and the rest of numbl. Variables stay in numbl's workspace between calls, as at the MATLAB prompt, so an array that does not change (a grid, say) needs to be sent only once.

## Functions

| Function                                          | Description                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `numbl_open()`                                    | Connect to a running plot server, or start one. Returns `NULL` on failure.                             |
| `numbl_put_matrix(e, name, data, rows, cols)`     | Queue a double matrix stored column-major: element `(i, j)` at `data[i + j*rows]`. The data is copied. |
| `numbl_put_matrix_f32(e, name, data, rows, cols)` | The same from `float` data (half the bytes to send).                                                   |
| `numbl_put_scalar(e, name, value)`                | Queue a scalar.                                                                                        |
| `numbl_eval(e, code)`                             | Send the queued variables and run `code`. Returns at once.                                             |
| `numbl_eval_sync(e, code)`                        | Wait for earlier frames, run `code`, and return 0, or -1 on error.                                     |
| `numbl_last_error(e)`                             | Message for the most recent error.                                                                     |
| `numbl_close(e)`                                  | Send what is still queued and disconnect.                                                              |

A C array declared as `a[rows][cols]` is row-major: pass it as a `cols x rows` matrix and transpose in MATLAB with `u.'`.

Text that the code displays goes to stdout and errors go to stderr. Call the functions of one engine from one thread at a time.

## How it works

`numbl_put_*` copies the arrays and `numbl_eval` queues them with the code as one frame; a background thread sends the frames to the plot server, so the simulation does not wait for the plot. If a frame is still waiting to be sent when a newer one with the same code arrives, the newer one replaces it. So when the simulation produces frames faster than the browser can draw them, the viewer shows the latest data and the frames in between are skipped. In the demo, plotting takes under 1% of the simulation's run time.

## Running the server separately

`numbl_open` starts `numbl plot-server` unless the `NUMBL_PLOT_URL` environment variable points to one that is already running. To keep one viewer open across runs:

```bash
numbl plot-server --plot-port 8790
NUMBL_PLOT_URL=http://127.0.0.1:8790/ ./heat2d
```

The server listens only on 127.0.0.1, since it runs the code it receives. For a simulation on a remote machine, run the server on your own computer and forward its port to the remote machine with ssh:

```bash
numbl plot-server --plot-port 8790        # on your computer
ssh -R 8790:127.0.0.1:8790 remote-host    # then, on the remote machine:
NUMBL_PLOT_URL=http://127.0.0.1:8790/ ./heat2d
```

`NUMBL_CMD` sets the command that `numbl_open` uses to start the server (default `numbl`).

## Limitations

- Large images update slowly: a 1000 x 1000 `imagesc` reaches the browser a few times per second.
- 3-D plots can be rotated but not panned, and their color limits cannot be changed interactively.
- There is no Fortran interface yet.
- Windows is not supported.
