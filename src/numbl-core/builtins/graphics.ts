/**
 * Graphics built-in functions: figure, plot, hold
 */

import { register, builtinSingle } from "./registry.js";
import { RTV, toNumber } from "../runtime/index.js";
import { isRuntimeNumber, FloatXArray } from "../runtime/types.js";
import { IType } from "../lowering/itemTypes.js";

export function registerGraphicsFunctions(): void {
  // Placeholder registrations — real work is done via codegen / runtime special cases
  const placeholderNames = [
    "figure",
    "plot",
    "plot3",
    "surf",
    "hold",
    "grid",
    "clf",
    "close",
    "title",
    "xlabel",
    "ylabel",
    "shading",
    "subplot",
    "legend",
    "sgtitle",
    "zlabel",
    "colorbar",
    "colormap",
    "axis",
    "view",
    "imagesc",
    "contour",
    "contourf",
    "mesh",
    "waterfall",
    "scatter",
    "drawnow",
    "pause",
  ];
  const placeholder = builtinSingle(() => RTV.num(0));
  for (const name of placeholderNames) {
    register(name, placeholder);
  }

  // ishold() — real work is done via $rt.ishold() in codegen
  register(
    "ishold",
    builtinSingle(() => RTV.logical(false), { outputType: IType.Logical })
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
