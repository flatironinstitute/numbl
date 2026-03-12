/**
 * Graphics built-in functions: figure, plot, hold
 */

import { register, builtinSingle } from "./registry.js";
import { RTV } from "../runtime/index.js";
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
}
