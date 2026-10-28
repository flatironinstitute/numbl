/**
 * Name resolution: what a name means where it is used, and the function
 * body a resolved call runs. See docs/developer_reference/compiler/
 * function-resolution.md.
 */

export { Scope } from "./scope.js";
export {
  resolveName,
  resolveQualified,
  targetOf,
  definingFile,
  type NameResolution,
  type ResolveOptions,
} from "./resolveName.js";
export {
  resolveFunction,
  resolutionArgType,
  type ResolvedTarget,
} from "./resolveFunction.js";
export {
  functionBodyFor,
  constructorBodyFor,
  oldStyleConstructorBodyFor,
  type FunctionBody,
} from "./functionBody.js";
export { resolveClassMethod } from "./classMethod.js";
