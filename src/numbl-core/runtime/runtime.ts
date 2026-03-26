/**
 * Simplified runtime providing the $rt interface for generated JS.
 *
 * This is a focused rewrite of runtime/runtime.ts (~3300 lines) that provides
 * exactly the methods the codegen emits, without class support, namespace
 * dispatch, WebGPU, subsref/subsasgn overloads, or property accessors.
 *
 * It reuses existing runtime value types, operations, indexing, and builtins.
 *
 * Operator functions are in runtimeOperators.ts, dispatch/registration in
 * runtimeDispatch.ts, and helper types/utilities in runtimeHelpers.ts.
 */

import {
  type RuntimeValue,
  type RuntimeLogical,
  type RuntimeTensor,
  type RuntimeStruct,
  type RuntimeString,
  type RuntimeFunction,
  RTV,
  toNumber,
  toBool,
  displayValue,
  shareRuntimeValue,
  RuntimeError,
  type CallFrame,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeClassInstance,
  isRuntimeCell,
  isRuntimeLogical,
  isRuntimeSparseMatrix,
  FloatXArray,
  RuntimeChar,
  RuntimeCell,
  type RuntimeComplexNumber,
} from "../runtime/types.js";
import { getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { type ItemType } from "../lowering/itemTypes.js";
import { BinaryOperation } from "../lowering/index.js";
import { getBuiltin, getAllBuiltinNames, getConstant } from "../builtins";
import {
  COLON_SENTINEL,
  END_SENTINEL,
  type PlotInstruction,
  type ExecOptions,
} from "../executor/types.js";
import type { FileIOAdapter } from "../fileIOAdapter.js";
import { ensureRuntimeValue, type CallSite } from "./runtimeHelpers.js";
import {
  uplus as _uplus,
  uminus as _uminus,
  transpose as _transpose,
  ctranspose as _ctranspose,
  not as _not,
  binop as _binop,
  range as _range,
  forIter as _forIter,
  emptyTensor as _emptyTensor,
  emptyStruct as _emptyStruct,
  makeCell as _makeCell,
  opHorzcat,
  opVertcat,
  switchMatch as _switchMatch,
} from "./runtimeOperators.js";
import {
  getFuncHandle as _getFuncHandle,
  makeUserFuncHandle as _makeUserFuncHandle,
  isa as _isa,
  callSuperConstructor as _callSuperConstructor,
  createClassInstance as _createClassInstance,
  dispatch as _dispatch,
  methodDispatch as _methodDispatch,
  callBuiltin as _callBuiltin,
  callClassMethod as _callClassMethod,
  numblClass as _numblClass,
} from "./runtimeDispatch.js";
import {
  index as _index,
  indexCell as _indexCell,
  indexStore as _indexStore,
  indexCellStore as _indexCellStore,
  multiOutputCellAssign as _multiOutputCellAssign,
} from "./runtimeIndexing.js";
import { registerSpecialBuiltins } from "./specialBuiltins.js";
import {
  getMember as _getMember,
  getMemberDynamic as _getMemberDynamic,
  getMemberOrEmpty as _getMemberOrEmpty,
  setMemberReturn as _setMemberReturn,
  setMemberDynamicReturn as _setMemberDynamicReturn,
  subsrefCall as _subsrefCall,
  subsasgnCall as _subsasgnCall,
  memberChainAssign as _memberChainAssign,
} from "./runtimeMemberAccess.js";
import {
  plotInstr as _plotInstr,
  plotCall as _plotCall,
  plot3Call as _plot3Call,
  surfCall as _surfCall,
  scatterCall as _scatterCall,
  imagescCall as _imagescCall,
  contourCall as _contourCall,
  meshCall as _meshCall,
  viewCall as _viewCall,
  legendCall as _legendCall,
  drawnow as _drawnow,
  pause as _pause,
} from "./runtimePlot.js";
import { isRuntimeChar, isRuntimeString, kstr } from "./types.js";
import { toString as _toString } from "./convert.js";

// ── Runtime class ────────────────────────────────────────────────────

export class Runtime {
  outputLines: string[] = [];
  plotInstructions: PlotInstruction[] = [];
  variableValues: Record<string, RuntimeValue> = {};
  holdState = false;

  // Line tracking
  public $line = 0;
  public $file: string | null = null;
  public $callStack: CallFrame[] = [];

  // Sentinel values
  public COLON = COLON_SENTINEL;
  public END = END_SENTINEL;

  // Constructor helpers
  public RTV = RTV;

  // Persistent variable storage: funcJsId → varName → value
  private persistentStore = new Map<string, Map<string, RuntimeValue>>();

  // Global variable storage: varName → value (shared across all functions)
  // Public object so generated code can read/write as $rt.$g["name"]
  public $g: Record<string, RuntimeValue> = {};

  // Accessor guard: prevents recursive getter/setter/subsref calls
  public activeAccessors = new Set<string>();

  // Callbacks into the compiler for class information (set by executeCode).
  // resolveClassMethod compiles and returns a class method on demand.
  public resolveClassMethod:
    | ((
        className: string,
        methodName: string
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ) => ((...args: any[]) => any) | null)
    | null = null;
  // getClassParent returns the parent class name (for isa / inheritance).
  public getClassParent: ((className: string) => string | null) | null = null;
  // Cache for resolved class methods: "className.methodName" → fn | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public classMethodCache = new Map<string, ((...args: any[]) => any) | null>();

  // Workspace accessors: varName → { get, set } closures over script-level vars
  // Registered by generated code so assignin/evalin can access workspace variables
  public workspaceAccessors = new Map<
    string,
    { get: () => unknown; set: (v: unknown) => void }
  >();

  // Dynamic workspace variables: for vars NOT declared via % external-access
  public dynamicWorkspaceVars = new Map<string, unknown>();

  // Caller accessor stack for evalin/assignin('caller', ...)
  // Each entry is either null (no accessors needed) or a record of var accessors.
  // Pushed/popped by generated function code.
  public callerAccessorStack: (Record<
    string,
    [() => unknown, (v: unknown) => void]
  > | null)[] = [];

  // Dynamic caller variable stack: parallel to callerAccessorStack
  // Each entry is a Map for dynamic vars (or null, lazily allocated)
  private dynamicCallerVarStack: (Map<string, unknown> | null)[] = [];

  // File I/O adapter (injected from host environment)
  public fileIO?: FileIOAdapter;

  // Builtin wrappers
  public builtins: Record<
    string,
    (nargout: number, args: unknown[]) => unknown
  > = {};

  // Custom builtins (execution-specific overrides, e.g. mip load's addpath).
  // These take priority over IBuiltins.
  public customBuiltins: Record<
    string,
    (nargout: number, args: unknown[]) => unknown
  > = {};

  // JIT compilation callback: compiles and evaluates a specialized function at
  // runtime. Set by executeCode() to close over the LoweringContext and Codegen.
  // The callback handles the full resolution chain (local → class method →
  // workspace) internally, using callSite for context. Returns the callable
  // function or null if nothing was found.
  public compileSpecialized:
    | ((
        name: string,
        argTypes: ItemType[],
        callSite: CallSite
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ) => ((...args: any[]) => any) | null)
    | null = null;

  // Callback for eval-with-local-variables. Set by executeCode to avoid circular imports.
  public evalLocalCallback:
    | ((
        code: string,
        initialVars: Record<string, RuntimeValue>,
        onOutput: (text: string) => void
      ) => {
        returnValue: unknown;
        variableValues: Record<string, RuntimeValue>;
      })
    | null = null;

  // Per-function JIT-compiled code for debugging: jsId → generated JS
  public jitFunctionCode = new Map<string, string>();

  // Cache for JIT-compiled functions: cacheKey → JS function
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private dispatchUnknownCache = new Map<string, (...args: any[]) => any>();

  // Always-on counter: how many times dispatchUnknown was called per function name
  public dispatchUnknownCounts = new Map<string, number>();

  // Profiling: stack-based self-time profiler for disjoint timings
  public profilingEnabled = false;
  private profileStack: { key: string; startTime: number }[] = [];
  private profileAccum = new Map<string, number>();
  private profileCounts = new Map<string, number>();

  constructor(
    private options: ExecOptions,
    private initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.profilingEnabled = !!options.profile;
    this.fileIO = options.fileIO;
    if (options.initialHoldState) {
      this.holdState = options.initialHoldState;
    }
    this.initBuiltins();
  }

  // ── Builtin initialization ──────────────────────────────────────────

  private initBuiltins(): void {
    for (const name of getAllBuiltinNames()) {
      const builtin = getBuiltin(name)!;
      const singleBranch = builtin.length === 1 ? builtin[0] : null;
      this.builtins[name] = (nargout: number, args: unknown[]) => {
        const margs = args.map(a => ensureRuntimeValue(a));
        let branch: (typeof builtin)[0];
        if (singleBranch) {
          branch = singleBranch;
        } else {
          const argItemTypes = margs.map(arg =>
            getItemTypeFromRuntimeValue(arg)
          );
          branch = builtin[0];
          for (let i = 0; i < builtin.length; i++) {
            if (builtin[i].check(argItemTypes, nargout)) {
              branch = builtin[i];
              break;
            }
          }
        }
        if (this.profilingEnabled) {
          this.profileEnter("builtin:fallback:" + name);
          const result = branch.apply(margs, nargout);
          this.profileLeave();
          return this.unwrapResult(
            result as RuntimeValue | RuntimeValue[] | undefined
          );
        }
        return this.unwrapResult(
          branch.apply(margs, nargout) as
            | RuntimeValue
            | RuntimeValue[]
            | undefined
        );
      };
    }

    // Register special builtins
    registerSpecialBuiltins(this);

    // Register plot and surf as builtins so they can be called via $rt.builtins["plot"]/$rt.builtins["surf"]
    this.builtins["plot"] = (_nargout: number, args: unknown[]) => {
      this.plot_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["plot3"] = (_nargout: number, args: unknown[]) => {
      this.plot3_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["surf"] = (_nargout: number, args: unknown[]) => {
      this.surf_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["scatter"] = (_nargout: number, args: unknown[]) => {
      this.scatter_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["imagesc"] = (_nargout: number, args: unknown[]) => {
      this.imagesc_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["contour"] = (_nargout: number, args: unknown[]) => {
      this.contour_call(
        args.map(a => ensureRuntimeValue(a)),
        false
      );
    };
    this.builtins["contourf"] = (_nargout: number, args: unknown[]) => {
      this.contour_call(
        args.map(a => ensureRuntimeValue(a)),
        true
      );
    };
    this.builtins["mesh"] = (_nargout: number, args: unknown[]) => {
      this.mesh_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["waterfall"] = (_nargout: number, args: unknown[]) => {
      this.mesh_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["colormap"] = (_nargout: number, args: unknown[]) => {
      if (args.length > 0) {
        const rv = ensureRuntimeValue(args[0]);
        const name = _toString(rv).replace(/^"|"$/g, "");
        _plotInstr(this.plotInstructions, { type: "set_colormap", name });
      }
    };
    this.builtins["view"] = (_nargout: number, args: unknown[]) => {
      this.view_call(args.map(a => ensureRuntimeValue(a)));
    };
    this.builtins["zlabel"] = (_nargout: number, args: unknown[]) => {
      if (args.length > 0) {
        _plotInstr(this.plotInstructions, {
          type: "set_zlabel",
          text: args[0],
        });
      }
    };
    this.builtins["colorbar"] = (_nargout: number, args: unknown[]) => {
      const val =
        args.length > 0 ? _toString(ensureRuntimeValue(args[0])) : "on";
      _plotInstr(this.plotInstructions, { type: "set_colorbar", value: val });
    };
    this.builtins["axis"] = (_nargout: number, args: unknown[]) => {
      if (args.length > 0) {
        const val = _toString(ensureRuntimeValue(args[0])).replace(
          /^"|"$/g,
          ""
        );
        _plotInstr(this.plotInstructions, { type: "set_axis", value: val });
      }
    };
  }

  public profileEnter(key: string): void {
    if (!this.profilingEnabled) return;
    const now = performance.now();
    const stack = this.profileStack;
    if (stack.length > 0) {
      const top = stack[stack.length - 1];
      this.profileAccum.set(
        top.key,
        (this.profileAccum.get(top.key) ?? 0) + (now - top.startTime)
      );
    }
    stack.push({ key, startTime: now });
  }

  public profileLeave(): void {
    if (!this.profilingEnabled) return;
    const now = performance.now();
    const frame = this.profileStack.pop()!;
    this.profileAccum.set(
      frame.key,
      (this.profileAccum.get(frame.key) ?? 0) + (now - frame.startTime)
    );
    this.profileCounts.set(
      frame.key,
      (this.profileCounts.get(frame.key) ?? 0) + 1
    );
    if (this.profileStack.length > 0) {
      this.profileStack[this.profileStack.length - 1].startTime = now;
    }
  }

  public getBuiltinProfile(): Record<
    string,
    {
      fallback: { totalTimeMs: number; callCount: number };
      interp: { totalTimeMs: number; callCount: number };
      jit: { totalTimeMs: number; callCount: number };
    }
  > {
    const zero = () => ({ totalTimeMs: 0, callCount: 0 });
    const result: Record<
      string,
      {
        fallback: { totalTimeMs: number; callCount: number };
        interp: { totalTimeMs: number; callCount: number };
        jit: { totalTimeMs: number; callCount: number };
      }
    > = {};
    const prefix = "builtin:";
    for (const [key, time] of this.profileAccum) {
      if (!key.startsWith(prefix)) continue;
      const rest = key.slice(prefix.length);
      let source: "fallback" | "interp" | "jit";
      let name: string;
      if (rest.startsWith("fallback:")) {
        source = "fallback";
        name = rest.slice(9);
      } else if (rest.startsWith("interp:")) {
        source = "interp";
        name = rest.slice(7);
      } else if (rest.startsWith("jit:")) {
        source = "jit";
        name = rest.slice(4);
      } else {
        continue;
      }
      const count = this.profileCounts.get(key) ?? 0;
      if (count > 0) {
        if (!result[name])
          result[name] = { fallback: zero(), interp: zero(), jit: zero() };
        result[name][source] = { totalTimeMs: time, callCount: count };
      }
    }
    return result;
  }

  public getDispatchProfile(): Record<
    string,
    { totalTimeMs: number; callCount: number }
  > {
    const result: Record<string, { totalTimeMs: number; callCount: number }> =
      {};
    const prefix = "dispatch:";
    for (const [key, time] of this.profileAccum) {
      if (key.startsWith(prefix)) {
        const name = key.slice(prefix.length);
        const count = this.profileCounts.get(key) ?? 0;
        if (count > 0) {
          result[name] = { totalTimeMs: time, callCount: count };
        }
      }
    }
    return result;
  }

  public getJitCompileTimeMs(): number {
    return this.profileAccum.get("jit") ?? 0;
  }

  public getDispatchUnknownCounts(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [name, count] of this.dispatchUnknownCounts) {
      result[name] = count;
    }
    return result;
  }

  private unwrapResult(
    result: RuntimeValue | RuntimeValue[] | undefined
  ): unknown {
    if (result === undefined) return undefined;
    if (Array.isArray(result)) {
      return result.map(r => {
        if (isRuntimeNumber(r)) return r;
        if (isRuntimeTensor(r) && r.data.length === 1 && !r.imag)
          return r.data[0];
        return r;
      });
    }
    if (isRuntimeNumber(result)) return result;
    if (isRuntimeTensor(result) && result.data.length === 1 && !result.imag)
      return result.data[0];
    return result;
  }

  // ── Output ──────────────────────────────────────────────────────────

  output(text: string): void {
    this.outputLines.push(text);
    this.options.onOutput?.(text);
  }

  // ── Error handling ──────────────────────────────────────────────────

  public error(message: string): RuntimeError {
    const err = new RuntimeError(message);
    if (this.$line > 0) err.line = this.$line;
    return err;
  }

  public annotateError(e: unknown): void {
    if (e instanceof RuntimeError && e.callStack === null) {
      e.callStack = [...this.$callStack];
      if (e.file === null && this.$file !== null) e.file = this.$file;
      if (e.line === null && this.$line > 0) e.line = this.$line;
    }
  }

  public wrapError(e: unknown): RuntimeStruct {
    if (e instanceof RuntimeError) {
      return RTV.struct(
        new Map([
          ["message", RTV.string(e.message)],
          ["identifier", RTV.string(e.identifier)],
        ])
      );
    }
    if (e instanceof Error) {
      return RTV.struct(
        new Map([
          ["message", RTV.string(e.message)],
          ["identifier", RTV.string("")],
        ])
      );
    }
    return RTV.struct(
      new Map([
        ["message", RTV.string(String(e))],
        ["identifier", RTV.string("")],
      ])
    );
  }

  // ── Call stack ──────────────────────────────────────────────────────

  public pushCallFrame(name: string): void {
    if (this.$callStack.length >= 500) {
      throw new RuntimeError("Maximum call stack depth exceeded");
    }
    this.$callStack.push({
      name,
      callerFile: this.$file,
      callerLine: this.$line,
    });
  }

  public popCallFrame(): void {
    this.$callStack.pop();
  }

  // ── Type conversion ─────────────────────────────────────────────────

  public toBool(v: unknown): boolean {
    if (typeof v === "number") return v !== 0;
    if (typeof v === "boolean") return v;
    return toBool(ensureRuntimeValue(v));
  }

  public toNumber(v: unknown): number {
    if (typeof v === "number") return v;
    return toNumber(ensureRuntimeValue(v));
  }

  public lg(v: boolean): RuntimeLogical {
    return RTV.logical(v);
  }

  /** Scalar power that returns complex for negative base with fractional exponent */
  public pow(a: number, b: number): number | RuntimeComplexNumber {
    const r = Math.pow(a, b);
    if (isNaN(r) && !isNaN(a) && !isNaN(b)) {
      // Negative base with fractional exponent → complex result
      const absA = Math.abs(a);
      const theta = a < 0 ? Math.PI : 0;
      const lnR = Math.log(absA);
      const newRe = b * lnR;
      const newIm = b * theta;
      const expR = Math.exp(newRe);
      const re = expR * Math.cos(newIm);
      const im = expR * Math.sin(newIm);
      if (Math.abs(im) < 1e-15) return re;
      return RTV.complex(re, im);
    }
    return r;
  }

  public share(v: unknown): RuntimeValue {
    if (
      typeof v === "number" ||
      typeof v === "boolean" ||
      typeof v === "string"
    )
      return v;
    if (v && typeof v === "object" && "kind" in v) {
      return shareRuntimeValue(v as RuntimeValue);
    }
    return v as RuntimeValue;
  }

  // ── Display ─────────────────────────────────────────────────────────

  public displayResult(v: unknown): void {
    if (!this.options.displayResults) return;
    if (v === undefined || v === null) return;
    const mv = ensureRuntimeValue(v);
    const text = displayValue(mv);
    this.output("ans =\n" + text + "\n\n");
  }

  public displayAssign(name: string, v: unknown): void {
    if (!this.options.displayResults) return;
    if (v === undefined || v === null) return;
    const mv = ensureRuntimeValue(v);
    const text = displayValue(mv);
    this.output(name + " =\n" + text + "\n\n");
  }

  // ── Constants ───────────────────────────────────────────────────────

  public getConstant(name: string): RuntimeValue {
    const c = getConstant(name);
    if (c === undefined) throw new RuntimeError(`Undefined constant: ${name}`);
    return c;
  }

  // ── String literals ─────────────────────────────────────────────────

  public makeChar(raw: string): RuntimeChar {
    let s = raw.slice(1, raw.length - 1);
    s = s.replaceAll("''", "'");
    return RTV.char(s);
  }

  public makeString(raw: string): RuntimeString {
    let s = raw.slice(1, raw.length - 1);
    s = s.replaceAll('""', '"');
    return RTV.string(s);
  }

  // ── Variables ───────────────────────────────────────────────────────

  public getInitialVariableValue(name: string): RuntimeValue | undefined {
    return this.initialVariableValues
      ? this.initialVariableValues[name]
      : undefined;
  }

  public setVariableValue(name: string, value: RuntimeValue | undefined): void {
    if (value === undefined) return;
    this.variableValues[name] = ensureRuntimeValue(value);
  }

  // ── who() / whos() ─────────────────────────────────────────────────

  /** Collect defined variable names from getters and apply argument filters. */
  private filterVarNames(
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): { names: string[]; values: Map<string, unknown> } {
    const values = new Map<string, unknown>();
    for (const [name, getter] of Object.entries(getters)) {
      const v = getter();
      if (v !== undefined) values.set(name, v);
    }
    let names = [...values.keys()].sort();

    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length > 0) {
      const strArgs = margs.map(a => {
        if (isRuntimeChar(a)) return a.value;
        if (isRuntimeString(a)) return a;
        if (typeof a === "string") return a;
        return String(a);
      });

      if (strArgs[0] === "-regexp") {
        const regexps = strArgs.slice(1).map(s => new RegExp(s));
        names = names.filter(name => regexps.some(re => re.test(name)));
      } else {
        const patterns = strArgs.map(s => {
          const escaped = s
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*/g, ".*")
            .replace(/\?/g, ".");
          return new RegExp(`^${escaped}$`);
        });
        names = names.filter(name => patterns.some(re => re.test(name)));
      }
    }
    return { names, values };
  }

  /** Get shape of a runtime value as a number array. */
  private static varShape(v: unknown): number[] {
    const rv = ensureRuntimeValue(v);
    if (isRuntimeNumber(rv) || isRuntimeLogical(rv)) return [1, 1];
    if (isRuntimeTensor(rv))
      return rv.shape.length >= 2 ? [...rv.shape] : [1, ...rv.shape];
    if (isRuntimeSparseMatrix(rv)) return [rv.m, rv.n];
    if (isRuntimeCell(rv)) return [...rv.shape];
    if (isRuntimeChar(rv))
      return (
        rv.shape ?? (rv.value.length === 0 ? [0, 0] : [1, rv.value.length])
      );
    if (isRuntimeString(rv)) return [1, 1];
    if (isRuntimeClassInstance(rv)) return [1, 1];
    return [1, 1];
  }

  /** Estimate bytes for a runtime value. */
  private static varBytes(v: unknown): number {
    const rv = ensureRuntimeValue(v);
    if (isRuntimeNumber(rv)) return 8;
    if (isRuntimeLogical(rv)) return 1;
    if (isRuntimeTensor(rv)) {
      const n = rv.data.length;
      return n * 8 + (rv.imag ? n * 8 : 0);
    }
    if (isRuntimeSparseMatrix(rv)) {
      const nnz = rv.pr.length;
      // ir (int32) + pr (float64) per nnz, jc (int32) per column+1, optional pi
      return nnz * 4 + nnz * 8 + (rv.n + 1) * 4 + (rv.pi ? nnz * 8 : 0);
    }
    if (isRuntimeCell(rv)) return rv.data.length * 8;
    if (isRuntimeChar(rv)) return rv.value.length * 2;
    if (isRuntimeString(rv)) return rv.length * 2;
    return 0;
  }

  /** Check if a runtime value is complex. */
  private static varIsComplex(rv: RuntimeValue): boolean {
    if (isRuntimeTensor(rv)) return !!rv.imag;
    if (isRuntimeSparseMatrix(rv)) return !!rv.pi;
    return (
      typeof rv === "object" &&
      rv !== null &&
      "kind" in rv &&
      rv.kind === "complex_number"
    );
  }

  /** Check if a runtime value is sparse. */
  private static varIsSparse(rv: RuntimeValue): boolean {
    return isRuntimeSparseMatrix(rv);
  }

  public who(
    nargout: number,
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): unknown {
    const { names } = this.filterVarNames(getters, args);

    if (nargout >= 1) {
      return _makeCell(
        names.map(n => RTV.char(n)),
        [names.length, 1]
      );
    }
    if (names.length > 0) {
      this.output("Your variables are:\n\n");
      for (const name of names) {
        this.output(name + "  ");
      }
      this.output("\n\n");
    }
    return 0;
  }

  /** Build info record for a single variable. */
  private static varInfo(
    name: string,
    v: unknown
  ): {
    name: string;
    shape: number[];
    bytes: number;
    cls: string;
    complex: boolean;
    sparse: boolean;
  } {
    const rv = ensureRuntimeValue(v);
    return {
      name,
      shape: Runtime.varShape(v),
      bytes: Runtime.varBytes(v),
      cls: _numblClass(rv),
      complex: Runtime.varIsComplex(rv),
      sparse: Runtime.varIsSparse(rv),
    };
  }

  private static whosStruct(info: {
    name: string;
    shape: number[];
    bytes: number;
    cls: string;
    complex: boolean;
    sparse: boolean;
  }): RuntimeStruct {
    return RTV.struct(
      new Map<string, RuntimeValue>([
        ["name", RTV.char(info.name)],
        [
          "size",
          RTV.tensor(new FloatXArray(info.shape), [1, info.shape.length]),
        ],
        ["bytes", RTV.num(info.bytes)],
        ["class", RTV.char(info.cls)],
        ["global", RTV.logical(false)],
        ["sparse", RTV.logical(info.sparse)],
        ["complex", RTV.logical(info.complex)],
        [
          "nesting",
          RTV.struct(
            new Map<string, RuntimeValue>([
              ["function", RTV.char("")],
              ["level", RTV.num(0)],
            ])
          ),
        ],
        ["persistent", RTV.logical(false)],
      ])
    );
  }

  public whos(
    nargout: number,
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): unknown {
    const { names, values } = this.filterVarNames(getters, args);
    const infos = names.map(name => Runtime.varInfo(name, values.get(name)!));

    if (nargout >= 1) {
      if (infos.length === 0) {
        return Runtime.whosStruct({
          name: "",
          shape: [0, 0],
          bytes: 0,
          cls: "",
          complex: false,
          sparse: false,
        });
      }
      const structs = infos.map(info => Runtime.whosStruct(info));
      if (structs.length === 1) return structs[0];
      return RTV.structArray([...structs[0].fields.keys()], structs);
    }

    // Display mode
    if (infos.length === 0) return 0;

    const rows = infos.map(info => {
      const attrs: string[] = [];
      if (info.complex) attrs.push("complex");
      if (info.sparse) attrs.push("sparse");
      return {
        ...info,
        sizeStr: info.shape.join("x"),
        attrsStr: attrs.join(", "),
      };
    });

    const nameW = Math.max(4, ...rows.map(r => r.name.length));
    const sizeW = Math.max(4, ...rows.map(r => r.sizeStr.length));
    const bytesW = Math.max(5, ...rows.map(r => String(r.bytes).length));
    const classW = Math.max(5, ...rows.map(r => r.cls.length));
    const hasAttrs = rows.some(r => r.attrsStr.length > 0);
    const attrsHeader = hasAttrs ? "  Attributes" : "";

    this.output(
      `  ${"Name".padEnd(nameW)}  ${"Size".padStart(sizeW)}  ${"Bytes".padStart(bytesW)}  ${"Class".padEnd(classW)}${attrsHeader}\n\n`
    );
    for (const r of rows) {
      const attrsPart = hasAttrs ? `  ${r.attrsStr}` : "";
      this.output(
        `  ${r.name.padEnd(nameW)}  ${r.sizeStr.padStart(sizeW)}  ${String(r.bytes).padStart(bytesW)}  ${r.cls.padEnd(classW)}${attrsPart}\n`
      );
    }
    this.output("\n");
    return 0;
  }

  // ── Workspace accessors (for assignin/evalin) ──────────────────────

  public setWorkspaceAccessor(
    name: string,
    getter: () => unknown,
    setter: (v: unknown) => void
  ): void {
    this.workspaceAccessors.set(name, { get: getter, set: setter });
  }

  public getWorkspaceVariable(name: string): unknown | undefined {
    const accessor = this.workspaceAccessors.get(name);
    if (accessor) return accessor.get();
    // Dynamic fallback
    return this.dynamicWorkspaceVars.get(name);
  }

  public setWorkspaceVariable(name: string, value: unknown): void {
    const accessor = this.workspaceAccessors.get(name);
    if (accessor) {
      accessor.set(value);
    } else {
      // Dynamic fallback
      this.dynamicWorkspaceVars.set(name, value);
    }
  }

  // ── Caller accessors (for assignin/evalin('caller', ...)) ──────────

  public pushCallerAccessors(
    accessors: Record<string, [() => unknown, (v: unknown) => void]> | null
  ): void {
    this.callerAccessorStack.push(accessors);
    this.dynamicCallerVarStack.push(null); // lazily allocated
  }

  public popCallerAccessors(): void {
    this.callerAccessorStack.pop();
    this.dynamicCallerVarStack.pop();
  }

  public getCallerVariable(name: string): unknown | undefined {
    if (this.callerAccessorStack.length < 2) return undefined;
    // Check static accessors first
    const callerAccessors =
      this.callerAccessorStack[this.callerAccessorStack.length - 2];
    if (callerAccessors) {
      const accessor = callerAccessors[name];
      if (accessor) return accessor[0]();
    }
    // Dynamic fallback
    const dynamicFrame =
      this.dynamicCallerVarStack[this.dynamicCallerVarStack.length - 2];
    return dynamicFrame?.get(name);
  }

  public setCallerVariable(name: string, value: unknown): void {
    if (this.callerAccessorStack.length < 2) return;
    // Check static accessors first
    const callerAccessors =
      this.callerAccessorStack[this.callerAccessorStack.length - 2];
    if (callerAccessors) {
      const accessor = callerAccessors[name];
      if (accessor) {
        accessor[1](value);
        return;
      }
    }
    // Dynamic fallback
    let dynamicFrame =
      this.dynamicCallerVarStack[this.dynamicCallerVarStack.length - 2];
    if (!dynamicFrame) {
      dynamicFrame = new Map();
      this.dynamicCallerVarStack[this.dynamicCallerVarStack.length - 2] =
        dynamicFrame;
    }
    dynamicFrame.set(name, value);
  }

  // ── eval with local variable access ─────────────────────────────────

  public evalLocal(
    code: unknown,
    accessors: Record<string, [() => unknown, (v: unknown) => void]>
  ): unknown {
    const codeStr = _toString(ensureRuntimeValue(code));

    // Read current values from accessors
    const initialValues: Record<string, RuntimeValue> = {};
    for (const [name, [getter]] of Object.entries(accessors)) {
      const val = getter();
      if (val !== undefined && val !== null) {
        initialValues[name] = ensureRuntimeValue(val);
      }
    }

    if (!this.evalLocalCallback) {
      throw new RuntimeError(
        "eval: internal error - no eval callback available"
      );
    }

    const result = this.evalLocalCallback(codeStr, initialValues, text =>
      this.output(text)
    );

    // Write back modified variables
    for (const [name, value] of Object.entries(result.variableValues)) {
      const accessor = accessors[name];
      if (accessor) {
        accessor[1](value);
      }
    }

    return result.returnValue;
  }

  // ── Persistent variables ────────────────────────────────────────────

  public getPersistent(funcId: string, varName: string): RuntimeValue {
    const val = this.persistentStore.get(funcId)?.get(varName);
    if (val === undefined) return RTV.tensor(new FloatXArray(0), [0, 0]);
    return val;
  }

  public setPersistent(
    funcId: string,
    varName: string,
    value: RuntimeValue
  ): void {
    let funcMap = this.persistentStore.get(funcId);
    if (!funcMap) {
      funcMap = new Map();
      this.persistentStore.set(funcId, funcMap);
    }
    funcMap.set(varName, value);
  }

  // ── Thin wrappers to runtimeOperators ─────────────────────────────

  public uplus(v: unknown): RuntimeValue {
    return _uplus(v);
  }
  public uminus(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv)) return this.dispatch("uminus", 1, [v]);
    }
    return _uminus(v);
  }
  public transpose(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv)) return this.dispatch("transpose", 1, [v]);
    }
    return _transpose(v);
  }
  public ctranspose(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv))
        return this.dispatch("ctranspose", 1, [v]);
    }
    return _ctranspose(v);
  }
  public not(v: unknown): RuntimeLogical | RuntimeTensor {
    return _not(v);
  }
  private static readonly binopMethodMap: Record<string, string> = {
    [BinaryOperation.Add]: "plus",
    [BinaryOperation.Sub]: "minus",
    [BinaryOperation.Mul]: "mtimes",
    [BinaryOperation.Div]: "mrdivide",
    [BinaryOperation.Pow]: "mpower",
    [BinaryOperation.LeftDiv]: "mldivide",
    [BinaryOperation.ElemMul]: "times",
    [BinaryOperation.ElemDiv]: "rdivide",
    [BinaryOperation.ElemPow]: "power",
    [BinaryOperation.ElemLeftDiv]: "ldivide",
    [BinaryOperation.Equal]: "eq",
    [BinaryOperation.NotEqual]: "ne",
    [BinaryOperation.Less]: "lt",
    [BinaryOperation.LessEqual]: "le",
    [BinaryOperation.Greater]: "gt",
    [BinaryOperation.GreaterEqual]: "ge",
    [BinaryOperation.BitAnd]: "and",
    [BinaryOperation.BitOr]: "or",
  };
  public binop(op: string, a: unknown, b: unknown): unknown {
    // Class operator dispatch: if either operand is a class instance,
    // try to call the overloaded operator method
    if (typeof a !== "number" || typeof b !== "number") {
      const ma = typeof a === "number" ? null : ensureRuntimeValue(a);
      const mb = typeof b === "number" ? null : ensureRuntimeValue(b);
      if (
        (ma && isRuntimeClassInstance(ma)) ||
        (mb && isRuntimeClassInstance(mb))
      ) {
        const methodName = Runtime.binopMethodMap[op];
        if (methodName) {
          return this.dispatch(methodName, 1, [a, b]);
        }
      }
    }
    return _binop(op, a, b);
  }
  public range(start: unknown, step: unknown, end: unknown): unknown {
    return _range(start, step, end);
  }
  public forIter(v: unknown): unknown[] {
    return _forIter(v);
  }
  public emptyTensor(): RuntimeTensor {
    return _emptyTensor();
  }
  public emptyStruct(): RuntimeStruct {
    return _emptyStruct();
  }
  public makeCell(elems: unknown[], shape: number[]): RuntimeCell {
    return _makeCell(elems, shape);
  }
  /**
   * Build a struct from trailing name-value pair arguments.
   * @param nvArgs  The rest-parameter array of trailing name-value pairs
   * @param defaults  An object mapping field names to default values
   */
  public buildNameValueStruct(
    nvArgs: unknown[],
    defaults: Record<string, unknown>
  ): RuntimeStruct {
    const fields = new Map<string, RuntimeValue>();
    // Apply defaults first
    for (const [k, v] of Object.entries(defaults)) {
      fields.set(k, ensureRuntimeValue(v));
    }
    // Override with passed name-value pairs
    for (let i = 0; i < nvArgs.length; i += 2) {
      const rv = ensureRuntimeValue(nvArgs[i]);
      const key =
        isRuntimeChar(rv) || isRuntimeString(rv) ? rstr(rv) : String(rv);
      const val = ensureRuntimeValue(nvArgs[i + 1]);
      fields.set(key, val);
    }
    return RTV.struct(fields);
  }
  public horzcat(elems: unknown[]): unknown {
    // Flatten first, then check for class instances
    const flat: unknown[] = [];
    for (const e of elems) {
      if (Array.isArray(e)) flat.push(...e);
      else flat.push(e);
    }
    // Single element: just return it
    if (flat.length === 1) return flat[0];
    const mvals = flat.map(e =>
      typeof e === "number" ? null : ensureRuntimeValue(e)
    );
    if (mvals.some(v => v && isRuntimeClassInstance(v))) {
      return this.dispatch("horzcat", 1, flat);
    }
    return opHorzcat(elems);
  }
  public vertcat(rows: unknown[]): unknown {
    // Single element: just return it (vertcat(x) == x)
    if (rows.length === 1) return rows[0];
    const mvals = rows.map(r =>
      typeof r === "number" ? null : ensureRuntimeValue(r)
    );
    if (mvals.some(v => v && isRuntimeClassInstance(v))) {
      return this.dispatch("vertcat", 1, rows);
    }
    return opVertcat(rows);
  }
  public switchMatch(control: unknown, caseVal: unknown): boolean {
    return _switchMatch(control, caseVal);
  }

  // ── Thin wrappers to runtimeDispatch ──────────────────────────────

  public getFuncHandle(name: string): RuntimeFunction {
    return _getFuncHandle(name);
  }
  public makeUserFuncHandle(
    jsFn: (...args: unknown[]) => unknown,
    nargin?: number
  ): RuntimeFunction {
    return _makeUserFuncHandle(jsFn, nargin);
  }
  public isa(value: unknown, classNameArg: unknown): RuntimeLogical {
    return _isa(this, value, classNameArg);
  }
  public callSuperConstructor(
    target: unknown,
    superInstance: unknown
  ): unknown {
    return _callSuperConstructor(target, superInstance);
  }
  public createClassInstance(
    className: string,
    propertyNames: string[],
    defaults?: Record<string, unknown>,
    isHandleClass = false
  ): RuntimeValue {
    return _createClassInstance(
      this,
      className,
      propertyNames,
      defaults,
      isHandleClass
    );
  }
  /**
   * Resolve a class method by name, using cache + JIT compilation.
   * Returns the compiled method function or null if it doesn't exist.
   */
  public cachedResolveClassMethod(
    className: string,
    methodName: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const cacheKey = `${className}.${methodName}`;
    if (this.classMethodCache.has(cacheKey)) {
      return this.classMethodCache.get(cacheKey)!;
    }
    if (!this.resolveClassMethod) return null;
    const fn = this.resolveClassMethod(className, methodName);
    this.classMethodCache.set(cacheKey, fn);
    return fn;
  }

  /**
   * Get the parent class name, using the compiler callback.
   * Returns null if no parent or no callback registered.
   */
  public getClassParentName(className: string): string | null {
    return this.getClassParent?.(className) ?? null;
  }

  /** Get the class name from a runtime value (for dynamic static method dispatch). */
  public getClassName(value: unknown): string {
    const mv = ensureRuntimeValue(value);
    if (isRuntimeClassInstance(mv)) return mv.className;
    throw new RuntimeError("Expected a class instance");
  }
  public dispatch(name: string, nargout: number, args: unknown[]): unknown {
    return _dispatch(this, name, nargout, args);
  }
  public methodDispatch(
    name: string,
    nargout: number,
    args: unknown[]
  ): unknown {
    return _methodDispatch(this, name, nargout, args);
  }
  public callBuiltin(name: string, nargout: number, args: unknown[]): unknown {
    return _callBuiltin(this, name, nargout, args);
  }
  public callClassMethod(
    className: string,
    methodName: string,
    nargout: number,
    args: unknown[]
  ): unknown {
    return _callClassMethod(this, className, methodName, nargout, args);
  }

  /**
   * JIT dispatch for function calls where argument types were unknown at
   * compile time. Inspects runtime values to infer types, asks the compiler
   * to JIT-compile a specialized version, then calls it. The compiler
   * handles the full resolution chain (local → class method → workspace)
   * internally using the callSite for context.
   */
  public dispatchUnknown(
    name: string,
    nargout: number,
    args: unknown[],
    callSite: CallSite
  ): unknown {
    this.dispatchUnknownCounts.set(
      name,
      (this.dispatchUnknownCounts.get(name) ?? 0) + 1
    );
    // 1. Infer types from runtime values
    const argTypes: ItemType[] = args.map(a =>
      getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
    );

    // 2. Cache check
    const cacheKey = JSON.stringify({ name, callSite, argTypes });
    let fn = this.dispatchUnknownCache.get(cacheKey);
    if (fn) {
      if (this.profilingEnabled) {
        this.profileEnter("dispatch:" + name);
        const result = fn(nargout, ...args);
        this.profileLeave();
        return result;
      }
      return fn(nargout, ...args);
    }

    // 3. JIT compile — the callback handles the full resolution chain
    if (this.compileSpecialized) {
      if (this.profilingEnabled) this.profileEnter("jit");
      const compiled = this.compileSpecialized(name, argTypes, callSite);
      if (compiled) {
        fn = compiled;
        this.dispatchUnknownCache.set(cacheKey, fn);
        if (this.profilingEnabled) {
          this.profileLeave(); // leave jit
          this.profileEnter("dispatch:" + name);
          const result = fn(nargout, ...args);
          this.profileLeave(); // leave dispatch
          return result;
        }
        return fn(nargout, ...args);
      }
      if (this.profilingEnabled) this.profileLeave(); // leave jit
    }

    // 4. Builtin fallback
    return this.callBuiltin(name, nargout, args);
  }

  // ── Indexing ────────────────────────────────────────────────────────

  public index(
    base: unknown,
    indices: unknown[],
    nargout: number = 1,
    skipSubsref: boolean | string = false
  ): unknown {
    return _index(this, base, indices, nargout, skipSubsref);
  }

  public indexCell(base: unknown, indices: unknown[]): unknown {
    return _indexCell(this, base, indices);
  }

  /** Normalize a cell indexing result into an array for CSL spreading.
   *  If the result is already an array (multi-element indexing), return it as-is.
   *  If it's a single value (scalar indexing), wrap it in an array. */
  public asCsl(value: unknown): unknown[] {
    return Array.isArray(value) ? value : [value];
  }

  /** Like index() but returns an empty struct instead of throwing.
   *  Used for chained lvalue assignment: V(i).field = rhs where V may be uninitialized. */
  public indexOrEmpty(base: unknown, indices: unknown[]): unknown {
    if (base === undefined || base === null) return _emptyStruct();
    try {
      return _index(this, base, indices, 1);
    } catch {
      return _emptyStruct();
    }
  }

  /** Like indexOrEmpty() but uses built-in indexing (bypasses subsref) for
   *  class instances.  For scalar class instances, V(1) returns V itself.
   *  Used in V(i).field = rhs patterns where the () part should always use
   *  built-in array indexing, not the class's overloaded subsref. */
  public builtinIndexOrEmpty(base: unknown, indices: unknown[]): unknown {
    if (base === undefined || base === null) return _emptyStruct();
    try {
      return _index(this, base, indices, 1, true);
    } catch {
      return _emptyStruct();
    }
  }

  /** Like indexCell() but returns 0 instead of throwing.
   *  Used for chained lvalue assignment: data{i}.field = rhs where data may be uninitialized. */
  public indexCellOrEmpty(base: unknown, indices: unknown[]): unknown {
    try {
      return _indexCell(this, base, indices);
    } catch {
      return 0;
    }
  }

  public indexStore(
    base: unknown,
    indices: unknown[],
    rhs: unknown,
    skipSubsasgn = false
  ): unknown {
    return _indexStore(this, base, indices, rhs, skipSubsasgn);
  }

  /** Built-in index store for V(i).field = rhs patterns.
   *  For class instances, the element was already modified via setMemberReturn
   *  so we just return the new value.  For other types, delegate to indexStore. */
  public builtinIndexStore(
    base: unknown,
    indices: unknown[],
    value: unknown
  ): unknown {
    const mv = ensureRuntimeValue(base);
    if (isRuntimeClassInstance(mv)) {
      return value;
    }
    return _indexStore(this, base, indices, value);
  }

  /** Check if a value is a class instance of the given class name.
   *  Used by codegen for runtime skipSubsasgn checks on Unknown-type variables. */
  public isClassInstance(value: unknown, className: string): boolean {
    const mv = ensureRuntimeValue(value);
    return isRuntimeClassInstance(mv) && mv.className === className;
  }

  public indexCellStore(
    base: unknown,
    indices: unknown[],
    rhs: unknown
  ): unknown {
    return _indexCellStore(base, indices, rhs);
  }

  public multiOutputCellAssign(
    base: unknown,
    indices: unknown,
    results: unknown[]
  ): unknown {
    return _multiOutputCellAssign(base, indices, results);
  }

  // ── Member access ───────────────────────────────────────────────────

  public getMember(base: unknown, name: string): unknown {
    return _getMember(this, base, name);
  }

  public getMemberDynamic(base: unknown, nameExpr: unknown): RuntimeValue {
    return _getMemberDynamic(base, nameExpr);
  }

  public getMemberOrEmpty(base: unknown, name: string): RuntimeValue {
    return _getMemberOrEmpty(base, name);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public setMember(_base: unknown, _name: string, _rhs: unknown): void {
    throw this.error("Cannot assign to member of temporary expression");
  }

  public setMemberReturn(base: unknown, name: string, rhs: unknown): unknown {
    return _setMemberReturn(this, base, name, rhs);
  }

  public setMemberDynamicReturn(
    base: unknown,
    nameExpr: unknown,
    rhs: unknown
  ): RuntimeValue {
    return _setMemberDynamicReturn(base, nameExpr, rhs);
  }

  public subsrefCall(base: unknown, names: string[]): unknown {
    return _subsrefCall(this, base, names);
  }

  public subsasgnCall(base: unknown, names: string[], rhs: unknown): unknown {
    return _subsasgnCall(this, base, names, rhs);
  }

  public memberChainAssign(
    base: unknown,
    names: string[],
    rhs: unknown
  ): unknown {
    return _memberChainAssign(this, base, names, rhs);
  }

  // ── narginchk / nargoutchk ──────────────────────────────────────────

  public narginchk(
    actualNargin: unknown,
    minArgs: unknown,
    maxArgs: unknown
  ): void {
    const n =
      typeof actualNargin === "number"
        ? actualNargin
        : toNumber(actualNargin as RuntimeValue);
    const lo =
      typeof minArgs === "number" ? minArgs : toNumber(minArgs as RuntimeValue);
    const hi =
      typeof maxArgs === "number" ? maxArgs : toNumber(maxArgs as RuntimeValue);
    if (n < lo) throw new RuntimeError("Not enough input arguments.");
    if (n > hi) throw new RuntimeError("Too many input arguments.");
  }

  public nargoutchk(
    actualNargout: unknown,
    minArgs: unknown,
    maxArgs: unknown
  ): void {
    const n =
      typeof actualNargout === "number"
        ? actualNargout
        : toNumber(actualNargout as RuntimeValue);
    const lo =
      typeof minArgs === "number" ? minArgs : toNumber(minArgs as RuntimeValue);
    const hi =
      typeof maxArgs === "number" ? maxArgs : toNumber(maxArgs as RuntimeValue);
    if (n < lo) throw new RuntimeError("Not enough output arguments.");
    if (n > hi) throw new RuntimeError("Too many output arguments.");
  }

  // ── Plot instructions ───────────────────────────────────────────────

  public plot_instr(
    instr:
      | { type: "set_figure_handle"; handle: unknown }
      | { type: "plot"; x: unknown; y: unknown }
      | { type: "set_hold"; value: unknown }
      | { type: "close" }
      | { type: "close_all" }
      | { type: "clf" }
      | { type: "set_subplot"; rows: unknown; cols: unknown; index: unknown }
      | { type: "set_sgtitle"; text: unknown }
      | { type: "set_grid"; value: unknown }
  ): void {
    _plotInstr(this.plotInstructions, instr);
    if (instr.type === "set_hold") {
      // Track the hold state so ishold() can query it.
      // runtimePlot already parsed the value; read it from the last instruction.
      const last = this.plotInstructions[this.plotInstructions.length - 1];
      if (last && last.type === "set_hold") {
        this.holdState = last.value;
      }
    }
  }

  public ishold(): RuntimeValue {
    return RTV.logical(this.holdState);
  }

  public plot_call(args: RuntimeValue[]): void {
    _plotCall(this.plotInstructions, args);
  }

  public plot3_call(args: RuntimeValue[]): void {
    _plot3Call(this.plotInstructions, args);
  }

  public surf_call(args: RuntimeValue[]): void {
    _surfCall(this.plotInstructions, args);
  }

  public scatter_call(args: RuntimeValue[]): void {
    _scatterCall(this.plotInstructions, args);
  }

  public imagesc_call(args: RuntimeValue[]): void {
    _imagescCall(this.plotInstructions, args);
  }

  public contour_call(args: RuntimeValue[], filled: boolean): void {
    _contourCall(this.plotInstructions, args, filled);
  }

  public mesh_call(args: RuntimeValue[]): void {
    _meshCall(this.plotInstructions, args);
  }

  public view_call(args: RuntimeValue[]): void {
    _viewCall(this.plotInstructions, args);
  }

  public legend_call(args: RuntimeValue[]): void {
    _legendCall(this.plotInstructions, args);
  }

  // ── Drawnow / Pause ─────────────────────────────────────────────────

  public drawnow(): void {
    return _drawnow(this.plotInstructions, this.options);
  }

  public pause(seconds: unknown): void {
    return _pause(seconds);
  }
}

export const rstr = (s: RuntimeString | RuntimeChar): string => {
  if (isRuntimeString(s)) return s;
  if (isRuntimeChar(s)) return s.value;
  throw new RuntimeError(`Expected string or char, got ${kstr(s)}`);
};
