/**
 * ODE solvers (ode45, ode23) and deval, on the explicit Runge-Kutta core
 * in helpers/ode-rk.ts.
 */

import {
  type RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeFunction,
  isRuntimeNumber,
  isRuntimeStruct,
} from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { Runtime } from "../../runtime/runtime.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import {
  solveRK,
  dormandPrince45,
  bogackiShampine23,
  interpolateAtPoints,
  denseOutputEval,
  type RKTableau,
  type StepData,
} from "../../helpers/ode-rk.js";

/** Map sol structs to their dense output step data for deval. */
const _solStepData = new WeakMap<object, StepData[]>();

for (const [name, tableau] of [
  ["ode45", dormandPrince45],
  ["ode23", bogackiShampine23],
] as const) {
  builtin({
    name,
    nargout: 5,
    mfile: true,
    apply: (args, nargout, ctx) => _ode45Impl(ctx.rt, nargout, args, tableau),
  });
}

builtin({
  name: "deval",
  nargin: [2, 3],
  mfile: true,
  apply: args => _devalImpl(args),
});

// ── ode45 implementation ──────────────────────────────────────────────

function _ode45Impl(
  rt: Runtime,
  nargout: number,
  args: RuntimeValue[],
  tableau: RKTableau = dormandPrince45
): RuntimeValue | RuntimeValue[] {
  const solverName = tableau.name;
  if (args.length < 3)
    throw builtinError(
      "MATLAB:odearguments:TspanOrY0NotSupplied",
      `When the first argument to ${solverName} is a function handle, the tspan and y0 arguments must be supplied.`
    );

  // Parse odefun
  const odefun = args[0];
  if (!isRuntimeFunction(odefun))
    throw new RuntimeError(
      `${solverName}: first argument must be a function handle`
    );

  // Parse tspan
  const tspanRaw = args[1];
  let tspan: number[];
  if (isRuntimeNumber(tspanRaw)) {
    throw new RuntimeError(
      `${solverName}: tspan must be a vector with at least 2 elements`
    );
  } else if (isRuntimeTensor(tspanRaw)) {
    tspan = Array.from(tspanRaw.data);
  } else {
    throw new RuntimeError(`${solverName}: tspan must be a numeric vector`);
  }
  if (tspan.length < 2)
    throw new RuntimeError(
      `${solverName}: tspan must have at least 2 elements`
    );

  // Parse y0
  const y0Raw = args[2];
  let y0: number[];
  if (isRuntimeNumber(y0Raw)) {
    y0 = [y0Raw as number];
  } else if (isRuntimeTensor(y0Raw)) {
    y0 = Array.from(y0Raw.data);
  } else {
    throw new RuntimeError(`${solverName}: y0 must be a numeric vector`);
  }
  const neq = y0.length;

  // Parse options
  let relTol: number | undefined;
  let absTol: number | undefined;
  let maxStep: number | undefined;
  let initialStep: number | undefined;
  let eventsFn:
    | ((t: number, y: number[]) => [number[], boolean[], number[]])
    | undefined;

  if (args.length >= 4) {
    const optsRaw = args[3];
    if (isRuntimeStruct(optsRaw)) {
      const fields = optsRaw.fields;
      const rtField = fields.get("RelTol");
      if (rtField !== undefined) relTol = toNumber(rtField);
      const atField = fields.get("AbsTol");
      if (atField !== undefined) absTol = toNumber(atField);
      const msField = fields.get("MaxStep");
      if (msField !== undefined) maxStep = toNumber(msField);
      const isField = fields.get("InitialStep");
      if (isField !== undefined) initialStep = toNumber(isField);

      const evField = fields.get("Events");
      if (evField !== undefined) {
        if (!isRuntimeFunction(evField))
          throw new RuntimeError(
            `${solverName}: Events option must be a function handle`
          );
        const evHandle = evField;
        eventsFn = (
          t: number,
          y: number[]
        ): [number[], boolean[], number[]] => {
          const yTensor =
            neq === 1
              ? (y[0] as RuntimeValue)
              : RTV.tensor(allocFloat64Array(y), [neq, 1]);
          const result = rt.index(evHandle, [t as RuntimeValue, yTensor], 3);
          const resultArr = result as RuntimeValue[];

          const extractArray = (v: RuntimeValue): number[] => {
            if (isRuntimeNumber(v)) return [v as number];
            if (isRuntimeTensor(v)) return Array.from(v.data);
            return [toNumber(v)];
          };

          return [
            extractArray(ensureRuntimeValue(resultArr[0])),
            extractArray(ensureRuntimeValue(resultArr[1])).map(x => x !== 0),
            extractArray(ensureRuntimeValue(resultArr[2])),
          ];
        };
      }
    }
  }

  // Build the ODE function wrapper
  const odeFn = (t: number, y: number[]): number[] => {
    const yVal: RuntimeValue =
      neq === 1
        ? (y[0] as RuntimeValue)
        : RTV.tensor(allocFloat64Array(y), [neq, 1]);
    const resultRaw = rt.index(odefun, [t as RuntimeValue, yVal], 1);
    const result = ensureRuntimeValue(resultRaw as RuntimeValue);
    if (isRuntimeNumber(result)) return [result as number];
    if (isRuntimeTensor(result)) return Array.from(result.data);
    throw new RuntimeError(
      `${solverName}: odefun must return a numeric vector`
    );
  };

  // MATLAB default: MaxStep = 0.1 * span (scipy defaults to Infinity)
  const span = Math.abs(tspan[tspan.length - 1] - tspan[0]);
  const effectiveMaxStep = maxStep ?? 0.1 * span;

  // Solve
  const odeResult = solveRK(tableau, odeFn, tspan, y0, {
    relTol,
    absTol,
    maxStep: effectiveMaxStep,
    initialStep,
    events: eventsFn,
  });

  // If tspan has intermediate points, interpolate
  const useInterp = tspan.length > 2;
  let tVals: number[];
  let yVals: number[][];

  if (useInterp) {
    const interp = interpolateAtPoints(odeResult, tspan);
    tVals = interp.t;
    yVals = interp.y;
  } else {
    tVals = odeResult.t;
    yVals = odeResult.y;
  }

  const nPoints = tVals.length;

  // Build sol struct output (single output)
  if (nargout <= 1) {
    // sol.x = row vector of step boundary times
    const solSteps = odeResult.steps;
    const nSolPts = solSteps.length + 1;
    const xData = allocFloat64Array(nSolPts);
    xData[0] = solSteps.length > 0 ? solSteps[0].tOld : tspan[0];
    for (let j = 0; j < solSteps.length; j++) {
      xData[j + 1] = solSteps[j].tNew;
    }

    // sol.y = neq x nSolPts (each column is solution at a step boundary)
    const yData = allocFloat64Array(neq * nSolPts);
    const y0sol = solSteps.length > 0 ? solSteps[0].yOld : y0;
    for (let i = 0; i < neq; i++) yData[i] = y0sol[i];
    for (let j = 0; j < solSteps.length; j++) {
      for (let i = 0; i < neq; i++) {
        yData[(j + 1) * neq + i] = solSteps[j].yNew[i];
      }
    }

    const solFields: Record<string, RuntimeValue> = {
      x: RTV.tensor(xData, [1, nSolPts]),
      y: RTV.tensor(yData, [neq, nSolPts]),
      solver: RTV.char(solverName),
    };

    if (odeResult.te.length > 0) {
      solFields.xe = RTV.tensor(allocFloat64Array(odeResult.te), [
        1,
        odeResult.te.length,
      ]);
      const yeData = allocFloat64Array(neq * odeResult.ye.length);
      for (let j = 0; j < odeResult.ye.length; j++) {
        for (let i = 0; i < neq; i++) {
          yeData[j * neq + i] = odeResult.ye[j][i];
        }
      }
      solFields.ye = RTV.tensor(yeData, [neq, odeResult.ye.length]);
      solFields.ie = RTV.tensor(allocFloat64Array(odeResult.ie), [
        odeResult.ie.length,
        1,
      ]);
    }

    const solStruct = RTV.struct(solFields);
    _solStepData.set(solStruct, solSteps);
    return solStruct;
  }

  // Build [t, y] output (t is column vector, y is nPoints x neq matrix)
  const tData = allocFloat64Array(nPoints);
  // Column-major: y(i,j) at index j*nPoints + i
  const yData = allocFloat64Array(nPoints * neq);
  for (let j = 0; j < nPoints; j++) {
    tData[j] = tVals[j];
    for (let i = 0; i < neq; i++) {
      yData[i * nPoints + j] = yVals[j][i]; // column-major
    }
  }

  const tTensor = RTV.tensor(tData, [nPoints, 1]);
  const yTensor = RTV.tensor(yData, [nPoints, neq]);

  if (nargout <= 2) return [tTensor, yTensor];

  // [t, y, te, ye, ie] output
  const nEvents = odeResult.te.length;
  const teTensor =
    nEvents > 0
      ? RTV.tensor(allocFloat64Array(odeResult.te), [nEvents, 1])
      : RTV.tensor(allocFloat64Array(0), [0, 1]);

  let yeTensor: RuntimeValue;
  if (nEvents > 0) {
    const yeData2 = allocFloat64Array(nEvents * neq);
    for (let j = 0; j < nEvents; j++) {
      for (let i = 0; i < neq; i++) {
        yeData2[i * nEvents + j] = odeResult.ye[j][i];
      }
    }
    yeTensor = RTV.tensor(yeData2, [nEvents, neq]);
  } else {
    yeTensor = RTV.tensor(allocFloat64Array(0), [0, neq]);
  }

  const ieTensor =
    nEvents > 0
      ? RTV.tensor(allocFloat64Array(odeResult.ie), [nEvents, 1])
      : RTV.tensor(allocFloat64Array(0), [0, 1]);

  return [tTensor, yTensor, teTensor, yeTensor, ieTensor];
}

// ── deval implementation ──────────────────────────────────────────────

function _devalImpl(args: RuntimeValue[]): RuntimeValue {
  const sol = args[0];
  if (!isRuntimeStruct(sol))
    throw new RuntimeError(
      "deval: first argument must be a solution structure"
    );

  const xintRaw = args[1];
  let xint: number[];
  if (isRuntimeNumber(xintRaw)) {
    xint = [xintRaw as number];
  } else if (isRuntimeTensor(xintRaw)) {
    xint = Array.from(xintRaw.data);
  } else {
    throw new RuntimeError("deval: second argument must be a numeric vector");
  }

  // Retrieve dense output step data
  const steps = _solStepData.get(sol);
  if (!steps || steps.length === 0)
    throw new RuntimeError(
      "deval: solution structure has no dense output data"
    );

  const neq = steps[0].yOld.length;
  const nSteps = steps.length;
  const nPts = xint.length;
  const yData = allocFloat64Array(neq * nPts);

  for (let p = 0; p < nPts; p++) {
    const t = xint[p];

    // Find the step containing t
    let idx = 0;
    for (let s = 0; s < nSteps; s++) {
      const lo = Math.min(steps[s].tOld, steps[s].tNew);
      const hi = Math.max(steps[s].tOld, steps[s].tNew);
      if (t >= lo - 1e-14 && t <= hi + 1e-14) {
        idx = s;
        break;
      }
      idx = s;
    }
    if (idx >= nSteps) idx = nSteps - 1;

    const step = steps[idx];
    const x =
      Math.abs(step.h) < 1e-300
        ? 0
        : Math.max(0, Math.min(1, (t - step.tOld) / step.h));
    const yi = denseOutputEval(step.yOld, step.Q, step.h, x);

    // Column-major: y(i, p) at index p * neq + i
    for (let i = 0; i < neq; i++) {
      yData[p * neq + i] = yi[i];
    }
  }

  return RTV.tensor(yData, [neq, nPts]);
}
