# numbl figure HDF5 layout (v1)

numbl figures can be downloaded as a self-describing **HDF5** file (`.h5`)
containing everything needed to recreate the figure: all numeric data plus the
styling/layout metadata. The intent is portability — the file can be opened by
any standard HDF5 tool ([h5py](https://www.h5py.org/),
[h5web](https://github.com/silx-kit/h5web), Panoply, MATLAB `h5read`, R, Julia)
and uploaded to third-party services that understand HDF5.

The writer lives in [src/graphics/exportFigureHdf5.ts](../src/graphics/exportFigureHdf5.ts);
the schema version constant is in
[src/graphics/figureHdf5Schema.ts](../src/graphics/figureHdf5Schema.ts).

## Conventions

- **Datasets vs. attributes.** Numeric _data_ arrays (line `x`/`y`, surface
  grids, vertices, …) are stored as **datasets**, gzip-compressed. Styling and
  scalar metadata (colors, line widths, labels, limits, plot kind, …) are stored
  as **attributes** on the enclosing group.
- **Grids are row-major.** numbl stores matrices column-major (Fortran order).
  2-D grid fields (`surf`/`mesh`/`pcolor`/`contour`/`imagesc`/`heatmap`/`bar3`)
  are transposed to **row-major `[rows, cols]`** datasets so generic viewers
  render them correctly without knowing numbl's storage order.
- **NaN / Inf are native.** Non-finite values are stored as IEEE-754 floats
  (no sentinels), so missing/!finite samples round-trip exactly.
- **Compression.** All datasets use the gzip (DEFLATE) filter, level 4.
- **RGB colors** are length-3 `float64` attributes in `[0, 1]`.
- **Auto axis limits.** A `null` (auto) limit bound is encoded as `NaN` inside
  the `xlim`/`ylim`/`zlim` attribute pair.
- **Ragged face lists** (`patch` faces) are padded to a rectangular `int32`
  matrix with `-1` fill.

## Structure

```
/                                 (root)
  @numbl_figure_version : int     = 1
  @generator            : str     = "numbl"
  @sgtitle              : str     (optional, super-title)
  @subplot_rows         : int     (optional)
  @subplot_cols         : int     (optional)
  @current_axes         : int     (1-based current axes index)

/axes/<i>/                        (one group per axes; <i> is 1-based)
  @title @xlabel @ylabel @zlabel  : str
  @legend                         : str[]
  @colormap                       : str   (e.g. "parula")
  @axis_scale                     : str   ("linear"|"semilogx"|"semilogy"|"loglog")
  @axis_mode @shading @y_dir @colorbar_location : str
  @grid_on @box_on @hold_on @colorbar @axis_visible : int (0/1)
  @view_az @view_el               : float (3-D view angles)
  @caxis                          : float[2]
  @xlim @ylim @zlim               : float[2]  (NaN bound = auto)
  @area_base_value                : float
  colormap_data                   : dataset [N,3]  (custom colormap, optional)

/axes/<i>/traces/<k>/             (one group per trace, in draw order)
  @kind  : str   — see "Trace kinds" below
  <styling attributes…>          (color, lineStyle, marker, lineWidth, …)
  <data datasets…>               (x, y, z, …)
```

For a `uihtml` figure (MATLAB `uihtml`) there are no axes; instead:

```
/uihtml/  @id : str   @html : str   @data : str (optional, JSON)
```

## Trace kinds

The `@kind` attribute on each trace group identifies the plot type. Data fields
are datasets; everything else is an attribute.

| `kind`                    | data datasets                                                       | notable attributes                                        |
| ------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| `plot`, `line`, `area`    | `x`, `y`                                                            | `color`, `lineStyle`, `marker`, `lineWidth`, `markerSize` |
| `plot3`                   | `x`, `y`, `z`                                                       | as above                                                  |
| `surf`, `mesh`, `surface` | `x`, `y`, `z`, `c` `[rows,cols]`                                    | `faceColor`, `edgeColor`, `faceAlpha`, `rows`, `cols`     |
| `pcolor`                  | `x`, `y`, `c` `[rows,cols]`                                         | `edgeColor`, `faceAlpha`                                  |
| `contour`                 | `x`, `y`, `z` `[rows,cols]`                                         | `filled`, `levels`, `lineColor`                           |
| `imagesc`                 | `z` `[rows,cols]`, `x`/`y` (2-elem limits)                          | `rows`, `cols`                                            |
| `heatmap`                 | `data` `[rows,cols]`                                                | `xLabels`, `yLabels`                                      |
| `bar`, `barh`             | `x`, `y`                                                            | `color`, `width`                                          |
| `bar3`, `bar3h`           | `x`, `y`, `z` `[rows,cols]`                                         | `color`, `width`                                          |
| `errorbar`                | `x`, `y`, `yNeg`, `yPos`, `xNeg`, `xPos`                            | `color`, `lineStyle`                                      |
| `boxchart`                | `outliers`                                                          | `median`, `q1`, `q3`, `whiskerLow`, `whiskerHigh`, `x`    |
| `piechart`                | `values`, `colors` `[N,3]`                                          | `names`, `innerRadius`                                    |
| `quiver`, `quiver3`       | `x`, `y`, `z`, `u`, `v`, `w`                                        | `color`, `showArrowHead`                                  |
| `patch`                   | `vertices` `[N,D]`, `faces` `[N,M]` int (−1 pad), `faceVertexCData` | `faceColor`, `edgeColor`, `faceAlpha`                     |

(Field availability follows the trace interfaces in
[src/graphics/types.ts](../src/graphics/types.ts).)

## Reading example (h5py)

```python
import h5py
with h5py.File("figure_1.h5") as f:
    assert f.attrs["numbl_figure_version"] == 1
    ax = f["axes/1"]
    print("title:", ax.attrs.get("title"))
    for name, tr in ax["traces"].items():
        kind = tr.attrs["kind"]
        if kind == "plot":
            x, y = tr["x"][:], tr["y"][:]   # 1-D arrays, NaN preserved
        elif kind == "surf":
            z = tr["z"][:]                  # row-major [rows, cols]
```

## Versioning

The root `numbl_figure_version` attribute is bumped on incompatible schema
changes. Readers should check it before interpreting the layout.
