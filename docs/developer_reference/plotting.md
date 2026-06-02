# Plotting

## Instructions, not pixels

The interpreter's plotting commands (`plot`, `surf`, `imagesc`, `figure`, `hold`, etc.) do not render anything themselves. They emit serializable `PlotInstruction` objects — a structured description of the figure state — into an accumulator on the runtime. At flush points (`drawnow`, end of script, explicit refresh), the runtime hands the accumulated instructions to an `onDrawnow` callback provided by the platform.

This keeps numbl-core free of any rendering code and lets every platform surface render however it likes:

- **CLI** — serves the plot viewer bundle from a local HTTP server and pushes instructions to a connected browser.
- **Web app** — renders directly in-page using the same viewer bundle.
- **Server** — normally forwards instructions back to the calling client.

## The plot viewer

A separate Vite-built bundle that consumes `PlotInstruction` streams and draws to an HTML canvas. It owns:

- figure state management (multiple figures, active figure, hold semantics);
- 2-D plot rendering (lines, scatter, bars, images, contour, colorbar);
- 3-D plot rendering (surf, mesh, plot3);
- axes, labels, legends, and interaction (zoom, pan).

Because the viewer is its own bundle, it does not rebuild with the rest of numbl-core. **After changing anything under the graphics source tree, rebuild the viewer explicitly** — unit tests will not catch a stale bundle.

## Adding a plot function

The common shape is:

1. Add the command name to the interpreter's plot-command list (so it is recognized).
2. Emit an appropriate `PlotInstruction` from the implementation. Keep the instruction serializable (no functions, no class instances).
3. Handle the new instruction type in the viewer.
4. Rebuild the viewer bundle.
5. Add tests — preferably integration scripts that exercise the command end-to-end.

See `docs/adding-plot-functions.md` for the specific checklist.
