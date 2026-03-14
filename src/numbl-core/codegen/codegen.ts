/**
 * JavaScript code generation from IR.
 *
 * Generates JS that calls into the standard $rt runtime. Supports on-demand
 * function generation: when a local function call is encountered, the function
 * is lowered and codegen'd lazily.
 *
 * Expression generation is in codegenExpr.ts, statement generation in
 * codegenStmt.ts, helper utilities in codegenHelpers.ts, LValue assignment
 * in codegenLValue.ts, function definition in codegenFunction.ts, and
 * class-related generation in codegenClass.ts.
 */

import { typeToString, type ItemType } from "../lowering/itemTypes.js";
import { type LoweringContext } from "../lowering/loweringContext.js";
import {
  computeSpecKey,
  hashForJsId,
  JS_RESERVED,
  collectClassProperties,
  genPropertyDefaults,
} from "./codegenHelpers.js";
import {
  genExpr as _genExpr,
  genIndexArg as _genIndexArg,
} from "./codegenExpr.js";
import { genStmts as _genStmts, genStmt as _genStmt } from "./codegenStmt.js";
import { genLValueAssign as _genLValueAssign } from "./codegenLValue.js";
import {
  genFunctionDef as _genFunctionDef,
  emitReturnCapture as _emitReturnCapture,
} from "./codegenFunction.js";
import {
  ensureClassMethodGenerated as _ensureClassMethodGenerated,
  ensureClassRegistered as _ensureClassRegistered,
} from "./codegenClass.js";
import { IRVariable } from "../lowering/loweringTypes.js";
import { IRExpr, IRLValue, IRStmt } from "../lowering/nodes.js";
import { buildLineTable, offsetToLineFast } from "../runtime/index.js";

// ── Codegen ─────────────────────────────────────────────────────────────

export class Codegen {
  private lines: string[] = [];
  private indent = 0;
  private tempCounter = 0;

  // Function generation registry. Key: specKey (JSON string)
  public generatedFunctions = new Map<
    string,
    { functionId: string; jsId: string }
  >();

  // Tracks specializations currently being generated (for recursive call detection)
  // specKey → jsId (so we can emit forward references)
  private generationInProgress = new Map<string, string>();

  // VarId → JS name cache
  private varNameCache = new Map<string, string>();

  // Global variable overrides: VarId → $rt.$g["name"] expression.
  // Set by Global statement codegen so all reads/writes go through the shared store.
  public globalVarRefs = new Map<string, string>();

  // Stack of variable reference overrides (for closure capture in anonymous functions).
  // Each entry maps varId → replacement JS expression.
  private varRefOverrides: Map<string, string>[] = [];

  // Current function context stack (for return statement generation)
  public currentFunctionOutputs: Array<{
    outputs: { variable: IRVariable }[];
    resultVarName: string;
    hasVarargout: boolean;
    originalName: string;
    uninitializedOutputIds: Set<string>;
  }> = [];
  public currentFunctionJsId: string | null = null;
  public nargoutOverride: string | null = null;
  public insideDeferredEnd = false;

  // Builtin functions referenced by generated code.
  // Populated by useBuiltin(); declarations are emitted by getCode().
  private usedBuiltins = new Set<string>();

  // Two output buffers:
  //   definitions → all function definitions and registrations
  //   main        → script body / top-level code
  private definitionLines: string[] = [];

  // Emit capture stack: when non-empty, emit() writes to the top array
  // instead of the normal target buffer. Used by ensureGenerated to collect
  // each function's output atomically, preventing nested definitions from
  // interleaving with the outer function's body.
  private emitCaptureStack: string[][] = [];

  // Track which classes have been registered
  private registeredClasses = new Set<string>();

  // Per-function generated code (jsId → code string), for debugging
  public perFunctionCode = new Map<string, string>();

  // Line tracking: avoid emitting redundant $rt.$file and $rt.$line assignments.
  // Reset when entering a new emit capture (function definitions emit their own tracking).
  public lastEmittedFile: string | null = null;
  public lastEmittedLine: number | null = null;
  private savedLineTracking: Array<{
    file: string | null;
    line: number | null;
  }> = [];

  // Pre-built line tables for fast offset→line conversion (binary search).
  // Populated lazily per file on first use.
  private lineTables = new Map<string, number[]>();

  /** Get 1-based line number for a source offset, using a cached line table. */
  getLineForOffset(file: string, offset: number): number | null {
    let table = this.lineTables.get(file);
    if (!table) {
      const source = this.fileSources.get(file);
      if (!source) return null;
      table = buildLineTable(source);
      this.lineTables.set(file, table);
    }
    return offsetToLineFast(table, offset);
  }

  /**
   * Stack of variable ID sets shared from enclosing function scopes.
   * Used for nested functions: these vars should NOT be re-declared
   * by the nested function (they're shared by reference with the parent
   * via JavaScript closure).
   */
  sharedVarIdStack: Set<string>[] = [];

  /** Set of nested function names currently in scope (for direct call/handle resolution). */
  nestedFunctionNames = new Set<string>();

  /** When generating a class method, the method name (for per-file local function scoping). */
  currentMethodName: string | null = null;

  /** Extra offset for nargin in constructor functions (1 for constructors where obj is prepended). */
  narginAdjust = 0;

  /** When true, skip emitting $rt.$file / $rt.$line assignments. */
  public noLineTracking = false;

  constructor(
    public loweringCtx: LoweringContext,
    public fileSources: Map<string, string>
  ) {}

  // ── Emit ──────────────────────────────────────────────────────────

  emit(line: string): void {
    const prefix = "  ".repeat(this.indent);
    const formatted = prefix + line;
    if (this.emitCaptureStack.length > 0) {
      this.emitCaptureStack[this.emitCaptureStack.length - 1].push(formatted);
    } else {
      this.lines.push(formatted);
    }
  }

  pushIndent(): void {
    this.indent++;
  }

  popIndent(): void {
    this.indent--;
  }

  freshTemp(prefix = "$t"): string {
    return `${prefix}${this.tempCounter++}`;
  }

  varRef(id: string): string {
    // Check override stack first (top of stack takes priority)
    for (let i = this.varRefOverrides.length - 1; i >= 0; i--) {
      const override = this.varRefOverrides[i].get(id);
      if (override !== undefined) return override;
    }

    // Check global variable overrides
    const globalRef = this.globalVarRefs.get(id);
    if (globalRef !== undefined) return globalRef;

    const cached = this.varNameCache.get(id);
    if (cached) return cached;

    // Extract the name from the VarId (name_N format)
    const lastUnderscore = id.lastIndexOf("_");
    const name = lastUnderscore > 0 ? id.substring(0, lastUnderscore) : id;
    let jsName = name;
    if (JS_RESERVED.has(jsName) || jsName.startsWith("$")) {
      jsName = `$${jsName}`;
    }

    this.varNameCache.set(id, jsName);
    return jsName;
  }

  /** Push a variable reference override mapping (for closure capture). */
  pushVarRefOverride(mapping: Map<string, string>): void {
    this.varRefOverrides.push(mapping);
  }

  /** Pop the most recent variable reference override mapping. */
  popVarRefOverride(): void {
    this.varRefOverrides.pop();
  }

  sanitizeName(name: string): string {
    return name.replace(/[^a-zA-Z0-9_]/g, "$");
  }

  /**
   * Record a builtin function as used and return its hoisted JS variable name.
   * The actual declaration is emitted by getCode().
   */
  useBuiltin(name: string): string {
    this.usedBuiltins.add(name);
    return `$builtin_${name}`;
  }

  /** Check if a class inherits from handle (directly or transitively). */
  isHandleClass(classInfo: { superClass: string | null }): boolean {
    let parentName = classInfo.superClass;
    while (parentName) {
      if (parentName === "handle") return true;
      const parentInfo = this.loweringCtx.getClassInfo(parentName);
      if (!parentInfo) break;
      parentName = parentInfo.superClass;
    }
    return false;
  }

  getCode(): string {
    // Emit hoisted builtin lookups before everything else
    const builtinDecls: string[] = [];
    for (const name of [...this.usedBuiltins].sort()) {
      builtinDecls.push(
        `var $builtin_${name} = $rt.builtins[${JSON.stringify(name)}];`
      );
    }
    return [...builtinDecls, ...this.definitionLines, ...this.lines].join("\n");
  }

  /**
   * Push a new capture buffer. While active, all emit() calls write to this
   * buffer instead of the main output. Used by ensureGenerated to collect
   * each function definition atomically.
   */
  public pushEmitCapture(): void {
    this.emitCaptureStack.push([]);
    this.emitTargetSavedIndents.push(this.indent);
    this.indent = 0;
    // Save and reset line tracking so the captured function gets fresh tracking
    this.savedLineTracking.push({
      file: this.lastEmittedFile,
      line: this.lastEmittedLine,
    });
    this.lastEmittedFile = null;
    this.lastEmittedLine = null;
  }

  /**
   * Pop the current capture buffer and flush its lines to definitionLines.
   */
  private popEmitCaptureToDefinitions(): void {
    const captured = this.emitCaptureStack.pop()!;
    this.indent = this.emitTargetSavedIndents.pop()!;
    this.definitionLines.push(...captured);
    // Restore line tracking from before the capture
    const saved = this.savedLineTracking.pop()!;
    this.lastEmittedFile = saved.file;
    this.lastEmittedLine = saved.line;
  }

  private emitTargetSavedIndents: number[] = [];

  /**
   * Pop the current capture buffer and return its lines (without flushing).
   */
  popEmitCaptureLines(): string[] {
    const captured = this.emitCaptureStack.pop()!;
    this.indent = this.emitTargetSavedIndents.pop()!;
    // Restore line tracking from before the capture
    const saved = this.savedLineTracking.pop()!;
    this.lastEmittedFile = saved.file;
    this.lastEmittedLine = saved.line;
    return captured;
  }

  /**
   * Emit a pre-formatted line without adding indentation.
   */
  emitRaw(line: string): void {
    if (this.emitCaptureStack.length > 0) {
      this.emitCaptureStack[this.emitCaptureStack.length - 1].push(line);
    } else {
      this.lines.push(line);
    }
  }

  typeComment(ty: ItemType | undefined): string {
    if (!ty) return "";
    return ` /* ${typeToString(ty)} */`;
  }

  /**
   * Run a callback with temporary overrides to codegen context fields.
   * All overridden fields are restored in a finally block, ensuring
   * cleanup even if an exception is thrown.
   *
   * This is the single mechanism for all temporary state changes during
   * code generation. Use it instead of ad-hoc save/restore patterns.
   */
  withCodegenContext<T>(
    overrides: {
      loweringCtx?: LoweringContext;
      currentMethodName?: string | null;
      narginAdjust?: number;
      nargoutOverride?: string | null;
      insideDeferredEnd?: boolean;
      currentFunctionJsId?: string | null;
    },
    fn: () => T
  ): T {
    const saved = {
      loweringCtx: this.loweringCtx,
      currentMethodName: this.currentMethodName,
      narginAdjust: this.narginAdjust,
      nargoutOverride: this.nargoutOverride,
      insideDeferredEnd: this.insideDeferredEnd,
      currentFunctionJsId: this.currentFunctionJsId,
    };
    if (overrides.loweringCtx !== undefined)
      this.loweringCtx = overrides.loweringCtx;
    if (overrides.currentMethodName !== undefined)
      this.currentMethodName = overrides.currentMethodName;
    if (overrides.narginAdjust !== undefined)
      this.narginAdjust = overrides.narginAdjust;
    if (overrides.nargoutOverride !== undefined)
      this.nargoutOverride = overrides.nargoutOverride;
    if (overrides.insideDeferredEnd !== undefined)
      this.insideDeferredEnd = overrides.insideDeferredEnd;
    if (overrides.currentFunctionJsId !== undefined)
      this.currentFunctionJsId = overrides.currentFunctionJsId;
    try {
      return fn();
    } finally {
      this.loweringCtx = saved.loweringCtx;
      this.currentMethodName = saved.currentMethodName;
      this.narginAdjust = saved.narginAdjust;
      this.nargoutOverride = saved.nargoutOverride;
      this.insideDeferredEnd = saved.insideDeferredEnd;
      this.currentFunctionJsId = saved.currentFunctionJsId;
    }
  }

  // ── Thin wrappers to extracted modules ────────────────────────────

  genExpr(expr: IRExpr): string {
    return _genExpr(this, expr);
  }

  genStmt(stmt: IRStmt): void {
    _genStmt(this, stmt);
  }

  genStmts(stmts: IRStmt[]): void {
    _genStmts(this, stmts);
  }

  genIndexArg(expr: IRExpr): string {
    return _genIndexArg(this, expr);
  }

  // ── On-demand function generation ─────────────────────────────────

  /**
   * Shared bookkeeping for all ensure*Generated methods.
   * Handles cache check, in-progress tracking, and finalization.
   * The `lower` callback obtains the IR (including any pre-checks).
   * The `generate` callback handles codegen, registration, and context swapping.
   */
  ensureGenerated(opts: {
    specKey: string;
    jsId: string;
    lower: () => (IRStmt & { type: "Function" }) | null;
    generate: (funcIR: IRStmt & { type: "Function" }) => void;
  }): string | null {
    // Already fully generated?
    const existing = this.generatedFunctions.get(opts.specKey);
    if (existing) return existing.jsId;

    // Currently being generated? (recursive call — return jsId for forward reference)
    const inProgress = this.generationInProgress.get(opts.specKey);
    if (inProgress) return inProgress;

    // Mark as in-progress BEFORE lowering (so recursive calls find it)
    this.generationInProgress.set(opts.specKey, opts.jsId);

    const funcIR = opts.lower();
    if (!funcIR) {
      this.generationInProgress.delete(opts.specKey);
      return null;
    }

    this.pushEmitCapture();
    opts.generate(funcIR);
    // Capture per-function code before flushing to definitions
    const captured = this.emitCaptureStack[this.emitCaptureStack.length - 1];
    this.perFunctionCode.set(opts.jsId, captured.join("\n"));
    this.popEmitCaptureToDefinitions();

    // Move from in-progress to generated
    this.generationInProgress.delete(opts.specKey);
    this.generatedFunctions.set(opts.specKey, {
      functionId: funcIR.functionId,
      jsId: opts.jsId,
    });

    return opts.jsId;
  }

  /**
   * Ensure a specialized version of a local function has been generated.
   * Returns the jsId if generated (or in progress for recursive calls),
   * or null if it's not a local function.
   */
  ensureSpecializedFunctionGenerated(
    name: string,
    argTypes: ItemType[]
  ): string | null {
    // Include the file name in the spec key and jsId so that local helper
    // functions with the same name in different workspace files don't collide.
    // verify that none of the argTypes are unknown
    if (argTypes.some(type => type.kind === "Unknown")) {
      throw new Error(
        `Cannot generate specialized function for ${name}: unknown parameter types`
      );
    }
    const filePrefix = this.loweringCtx.getRelativePath(
      this.loweringCtx.mainFileName
    );
    const specKey = computeSpecKey(`${filePrefix}::${name}`, argTypes);
    const hash = hashForJsId(argTypes);
    const jsId = `$fn_${this.sanitizeName(filePrefix)}$${this.sanitizeName(name)}$${hash}`;

    return this.ensureGenerated({
      specKey,
      jsId,
      lower: () =>
        this.loweringCtx.getOrLowerFunctionSpecialized(name, argTypes),
      generate: funcIR => {
        this.genFunctionDef(funcIR, jsId);
      },
    });
  }
  /**
   * Ensure a specialized external function (workspace or private) has been generated.
   * Temporarily swaps the lowering context so the function body can
   * resolve its own local helpers (not the main file's).
   */
  private ensureExternalFunctionGenerated(
    name: string,
    argTypes: ItemType[],
    opts: {
      specKeyPrefix: string;
      jsIdPrefix: string;
      getContext: () => LoweringContext | null;
      lower: (argTypes: ItemType[]) => (IRStmt & { type: "Function" }) | null;
    }
  ): string | null {
    const specKey = computeSpecKey(opts.specKeyPrefix, argTypes);
    const hash = hashForJsId(argTypes);
    const jsId = `${opts.jsIdPrefix}${this.sanitizeName(name)}$${hash}`;
    let extCtx: LoweringContext | null = null;

    return this.ensureGenerated({
      specKey,
      jsId,
      lower: () => {
        extCtx = opts.getContext();
        if (!extCtx) return null;
        if (!this.fileSources.has(extCtx.mainFileName)) {
          this.fileSources.set(extCtx.mainFileName, extCtx.fileSource);
        }
        return opts.lower(argTypes);
      },
      generate: funcIR => {
        this.withCodegenContext({ loweringCtx: extCtx! }, () => {
          this.genFunctionDef(funcIR, jsId);
        });
      },
    });
  }

  ensureWorkspaceFunctionGenerated(
    name: string,
    argTypes: ItemType[]
  ): string | null {
    // verify that none of the argTypes are unknown
    if (argTypes.some(type => type.kind === "Unknown")) {
      throw new Error(
        `Cannot generate specialized function for ${name}: unknown parameter types`
      );
    }
    return this.ensureExternalFunctionGenerated(name, argTypes, {
      specKeyPrefix: name,
      jsIdPrefix: "$fn_",
      getContext: () => this.loweringCtx.getOrCreateWorkspaceFileContext(name),
      lower: at =>
        this.loweringCtx.getOrLowerWorkspaceFunctionSpecialized(name, at),
    });
  }

  ensurePrivateFunctionGenerated(
    name: string,
    argTypes: ItemType[]
  ): string | null {
    // verify that none of the argTypes are unknown
    if (argTypes.some(type => type.kind === "Unknown")) {
      throw new Error(
        `Cannot generate specialized function for ${name}: unknown parameter types`
      );
    }
    return this.ensureExternalFunctionGenerated(name, argTypes, {
      specKeyPrefix: `private:${name}`,
      jsIdPrefix: "$fn_priv_",
      getContext: () => this.loweringCtx.getOrCreatePrivateFileContext(name),
      lower: at =>
        this.loweringCtx.getOrLowerPrivateFunctionSpecialized(name, at),
    });
  }

  /**
   * Ensure a specialized class method (or constructor) has been generated.
   * Similar to ensureWorkspaceFunctionGenerated but uses the class file context.
   * When called from within a function body, redirects output to top-level scope.
   */
  ensureClassMethodGenerated(
    className: string,
    methodName: string,
    argTypes: ItemType[]
  ): string | null {
    return _ensureClassMethodGenerated(this, className, methodName, argTypes);
  }

  /**
   * Mark a class as registered (methods are generated on demand, not eagerly).
   */
  ensureClassRegistered(className: string): void {
    _ensureClassRegistered(this, className);
  }

  /**
   * Generate a constructor wrapper function for a workspace class.
   * Used by JIT compilation when `feval('ClassName')` is called at runtime.
   */
  ensureWorkspaceClassConstructorGenerated(
    className: string,
    argTypes: ItemType[]
  ): string | null {
    // verify that none of the argTypes are unknown
    if (argTypes.some(type => type && type.kind === "Unknown")) {
      throw new Error(
        `Cannot generate constructor for ${className}: unknown parameter types`
      );
    }
    const classInfo = this.loweringCtx.getClassInfo(className);
    if (!classInfo) return null;

    const specKey = computeSpecKey(`${className}$$ctor`, argTypes);
    const existing = this.generatedFunctions.get(specKey);
    if (existing) return existing.jsId;

    const hash = hashForJsId(argTypes);
    const jsId = `$fn_${this.sanitizeName(className)}$$ctor$${hash}`;

    // Collect properties from the full inheritance chain
    const {
      propertyNames: allPropertyNames,
      propertyDefaults: allPropertyDefaults,
    } = collectClassProperties(this.loweringCtx, className);
    const propsJson = JSON.stringify(allPropertyNames);
    const isHandleClass = this.isHandleClass(classInfo);
    const defaultsArg = genPropertyDefaults(
      this,
      className,
      allPropertyDefaults
    );

    // Generate constructor wrapper function
    this.pushEmitCapture();

    if (classInfo.constructorName) {
      const selfType: ItemType = { kind: "ClassInstance", className };
      const ctorArgTypes: ItemType[] = [selfType, ...argTypes];
      const ctorJsId = this.ensureClassMethodGenerated(
        className,
        classInfo.constructorName,
        ctorArgTypes
      );
      if (ctorJsId) {
        this.emit(`function ${jsId}($nargout, ...args) {`);
        this.pushIndent();
        this.emit(
          `var $inst = $rt.createClassInstance(${JSON.stringify(className)}, ${propsJson}, ${defaultsArg}, ${isHandleClass});`
        );
        this.emit(`return ${ctorJsId}(1, $inst, ...args);`);
        this.popIndent();
        this.emit(`}`);
      } else {
        this.emit(`function ${jsId}($nargout, ...args) {`);
        this.pushIndent();
        this.emit(
          `return $rt.createClassInstance(${JSON.stringify(className)}, ${propsJson}, ${defaultsArg}, ${isHandleClass});`
        );
        this.popIndent();
        this.emit(`}`);
      }
    } else {
      this.emit(`function ${jsId}($nargout, ...args) {`);
      this.pushIndent();
      this.emit(
        `return $rt.createClassInstance(${JSON.stringify(className)}, ${propsJson}, ${defaultsArg}, ${isHandleClass});`
      );
      this.popIndent();
      this.emit(`}`);
    }

    this.popEmitCaptureToDefinitions();

    this.generatedFunctions.set(specKey, {
      functionId: className,
      jsId,
    });
    return jsId;
  }

  // ── LValue assignment ─────────────────────────────────────────────

  genLValueAssign(lv: IRLValue, rhs: string): void {
    _genLValueAssign(this, lv, rhs);
  }

  // ── Function definition generation ────────────────────────────────

  genFunctionDef(stmt: IRStmt & { type: "Function" }, jsId?: string): void {
    _genFunctionDef(this, stmt, jsId);
  }

  emitReturnCapture(fnCtx: (typeof this.currentFunctionOutputs)[0]): void {
    _emitReturnCapture(this, fnCtx);
  }

  /** Clear all generated function cache entries for a given function name. */
  clearGeneratedFunction(name: string): void {
    for (const key of this.generatedFunctions.keys()) {
      const parsed = JSON.parse(key);
      if (parsed.name === name) {
        this.generatedFunctions.delete(key);
      }
    }
  }

  isClassRegistered(className: string): boolean {
    return this.registeredClasses.has(className);
  }

  markClassRegistered(className: string): void {
    this.registeredClasses.add(className);
  }
}
