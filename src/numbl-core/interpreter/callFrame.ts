/**
 * The interpreter's call frame: one activation of running code.
 */

import { Scope } from "../resolution/scope.js";
import type { CodeIndex } from "../workspace/codeIndex.js";
import type { Environment } from "./types.js";

/** Where code runs: a file, and the class and method whose code it is. A
 *  `Scope` and a `FunctionBody` are both code sites. */
export interface CodeSite {
  readonly file: string;
  readonly className?: string;
  readonly methodName?: string;
}

/**
 * One activation: the base workspace, a function call, or an anonymous
 * function call. A frame is the `Scope` names resolve from (its file, class,
 * method and environment), plus the frame it was called from and the
 * call-site names of its arguments.
 *
 * The interpreter's running frame (`interp.frame`) is replaced on entry to a
 * function body and restored on exit. Code that runs in the same activation
 * but comes from another file (a script called by name, `run`, a `.numbl.js`
 * function, a function handle) runs in a frame from `at(site)`, which shares
 * the environment, caller and input names.
 */
export class CallFrame extends Scope {
  declare readonly env: Environment;

  /** Argument names of the call expression this frame is evaluating, for
   *  the callee's `inputname`. Set by the interpreter around a call
   *  expression; a function that a builtin calls back while that call runs
   *  (`cellfun(h, c)`) sees them too, as in MATLAB. */
  callSiteNames: readonly string[] | undefined;

  constructor(
    code: CodeIndex,
    file: string,
    className: string | undefined,
    methodName: string | undefined,
    env: Environment,
    /** The frame this activation was called from (`evalin('caller')`,
     *  `assignin('caller')`); undefined for the base workspace. */
    readonly caller: CallFrame | undefined,
    /** Call-site variable names of the arguments, for `inputname`. Entry i
     *  is the name of the variable passed as argument i+1, or '' when that
     *  argument was not a plain variable. Undefined when the call did not
     *  come from an interpreted call expression. */
    readonly inputNames: readonly string[] | undefined,
    callSiteNames?: readonly string[]
  ) {
    super(code, file, className, methodName, env);
    this.callSiteNames = callSiteNames;
  }

  /** This activation running code from `site`. */
  at(site: CodeSite): CallFrame {
    if (
      site.file === this.file &&
      site.className === this.className &&
      site.methodName === this.methodName
    )
      return this;
    return new CallFrame(
      this.code,
      site.file,
      site.className,
      site.methodName,
      this.env,
      this.caller,
      this.inputNames,
      this.callSiteNames
    );
  }

  /** This activation evaluating text in `target`'s workspace (`evalin`):
   *  the variables and input names are `target`'s, while the code site and
   *  the caller stay this frame's, so a nested `evalin('caller')` inside
   *  `evalin('caller', ...)` reaches the same caller, as in MATLAB. */
  inWorkspaceOf(target: CallFrame): CallFrame {
    return new CallFrame(
      this.code,
      this.file,
      this.className,
      this.methodName,
      target.env,
      this.caller,
      target.inputNames
    );
  }

  /** This frame's site and environment without its caller chain or input
   *  names, for a function handle to keep as the scope it resolves names
   *  from. `env` replaces the environment (an anonymous function keeps a
   *  snapshot). (A detached frame rather than a plain `Scope` keeps the
   *  scopes that name resolution sees to one class.) */
  detach(env: Environment = this.env): CallFrame {
    return new CallFrame(
      this.code,
      this.file,
      this.className,
      this.methodName,
      env,
      undefined,
      undefined
    );
  }
}
