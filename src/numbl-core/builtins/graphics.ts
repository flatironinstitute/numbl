/**
 * Graphics built-in functions: figure, plot, hold
 */

import { register, builtinSingle } from "./registry.js";
import { RTV, toNumber } from "../runtime/index.js";
import { isRuntimeNumber, FloatXArray } from "../runtime/types.js";
import { IType } from "../lowering/itemTypes.js";

export function registerGraphicsFunctions(): void {
  // figure(handle) — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "figure",
    builtinSingle(() => RTV.num(0))
  );

  // plot(x, y) — placeholder; real work is done as a special case in runtime
  register(
    "plot",
    builtinSingle(() => RTV.num(0))
  );

  // plot3(X, Y, Z) — placeholder; real work is done as a special case in runtime
  register(
    "plot3",
    builtinSingle(() => RTV.num(0))
  );

  // surf(X, Y, Z) — placeholder; real work is done as a special case in runtime
  register(
    "surf",
    builtinSingle(() => RTV.num(0))
  );

  // hold('on') / hold('off') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "hold",
    builtinSingle(() => RTV.num(0))
  );

  // ishold() — real work is done via $rt.ishold() in codegen
  register(
    "ishold",
    builtinSingle(() => RTV.logical(false), { outputType: IType.Logical })
  );

  // grid('on') / grid('off') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "grid",
    builtinSingle(() => RTV.num(0))
  );

  // clf — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "clf",
    builtinSingle(() => RTV.num(0))
  );

  // close — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "close",
    builtinSingle(() => RTV.num(0))
  );

  // title('text') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "title",
    builtinSingle(() => RTV.num(0))
  );

  // xlabel('text') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "xlabel",
    builtinSingle(() => RTV.num(0))
  );

  // ylabel('text') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "ylabel",
    builtinSingle(() => RTV.num(0))
  );

  // shading(type) — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "shading",
    builtinSingle(() => RTV.num(0))
  );

  // subplot(rows, cols, index) — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "subplot",
    builtinSingle(() => RTV.num(0))
  );

  // legend(labels...) — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "legend",
    builtinSingle(() => RTV.num(0))
  );

  // sgtitle(text) — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "sgtitle",
    builtinSingle(() => RTV.num(0))
  );

  // zlabel('text') — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "zlabel",
    builtinSingle(() => RTV.num(0))
  );

  // colorbar — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "colorbar",
    builtinSingle(() => RTV.num(0))
  );

  // colormap — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "colormap",
    builtinSingle(() => RTV.num(0))
  );

  // axis — placeholder; real work is done via $rt.plot_instr in codegen
  register(
    "axis",
    builtinSingle(() => RTV.num(0))
  );

  // view — placeholder; real work is done via $rt.view_call in codegen
  register(
    "view",
    builtinSingle(() => RTV.num(0))
  );

  // imagesc — placeholder; real work is done as special case in runtime
  register(
    "imagesc",
    builtinSingle(() => RTV.num(0))
  );

  // contour — placeholder; real work is done as special case in runtime
  register(
    "contour",
    builtinSingle(() => RTV.num(0))
  );

  // contourf — placeholder; real work is done as special case in runtime
  register(
    "contourf",
    builtinSingle(() => RTV.num(0))
  );

  // mesh — placeholder; real work is done as special case in runtime
  register(
    "mesh",
    builtinSingle(() => RTV.num(0))
  );

  // waterfall — placeholder; real work is done as special case in runtime (renders as mesh)
  register(
    "waterfall",
    builtinSingle(() => RTV.num(0))
  );

  // scatter — placeholder; real work is done as special case in runtime
  register(
    "scatter",
    builtinSingle(() => RTV.num(0))
  );

  // drawnow
  register(
    "drawnow",
    builtinSingle(() => RTV.num(0))
  );

  // pause(seconds)
  register(
    "pause",
    builtinSingle(() => RTV.num(0))
  );

  // Colormap functions — return the colormap name as a char for use with colormap()
  const colormapNames = [
    "parula",
    "jet",
    "hsv",
    "hot",
    "cool",
    "spring",
    "summer",
    "autumn",
    "winter",
    "gray",
    "bone",
    "copper",
    "pink",
  ];
  for (const cm of colormapNames) {
    register(
      cm,
      builtinSingle(() => RTV.char(cm))
    );
  }

  // peaks(n) — generates the MATLAB peaks surface
  register(
    "peaks",
    builtinSingle(args => {
      const n =
        args.length > 0
          ? isRuntimeNumber(args[0])
            ? (args[0] as number)
            : toNumber(args[0])
          : 49;
      const data = new FloatXArray(n * n);
      for (let j = 0; j < n; j++) {
        const y = -3 + (6 * j) / (n - 1);
        for (let i = 0; i < n; i++) {
          const x = -3 + (6 * i) / (n - 1);
          const x2 = x * x;
          const y2 = y * y;
          const z =
            3 * Math.pow(1 - x, 2) * Math.exp(-x2 - Math.pow(y + 1, 2)) -
            10 * (x / 5 - x * x2 - Math.pow(y, 5)) * Math.exp(-x2 - y2) -
            (1 / 3) * Math.exp(-Math.pow(x + 1, 2) - y2);
          data[j * n + i] = z; // column-major: column j, row i
        }
      }
      return RTV.tensor(data, [n, n]);
    })
  );
}
