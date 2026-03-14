/**
 * LValue assignment code generation.
 *
 * Handles generating JS code for assigning to various LValue types:
 * variables, member access, dynamic member access, and indexing.
 */

import {
  type IRExpr,
  type IRLValue,
  itemTypeForExprKind,
} from "../lowering/index.js";
import type { Codegen } from "./codegen.js";
import { hasClassMethod } from "./codegenExpr.js";

/**
 * Generate JS code for an LValue assignment.
 */
export function genLValueAssign(cg: Codegen, lv: IRLValue, rhs: string): void {
  switch (lv.type) {
    case "Var":
      cg.emit(`${cg.varRef(lv.variable.id.id)} = ${rhs};`);
      break;

    case "Member":
      genMemberAssign(cg, lv, rhs);
      break;

    case "MemberDynamic":
      genMemberDynamicAssign(cg, lv, rhs);
      break;

    case "Index":
    case "IndexCell":
      genIndexAssign(cg, lv, rhs);
      break;
  }
}

// ── General chain helpers ─────────────────────────────────────────────

/** A single step in a flattened access chain. */
type ChainStep =
  | { type: "member"; name: string }
  | { type: "index"; indices: string[]; isCell: boolean };

/**
 * Flatten an expression into a linear sequence of access steps and a root.
 * Walks through Member, Index, and IndexCell nodes recursively.
 */
function flattenExprChain(
  cg: Codegen,
  expr: IRExpr
): { steps: ChainStep[]; root: IRExpr } {
  const steps: ChainStep[] = [];
  let cursor = expr;
  while (true) {
    if (cursor.kind.type === "Member") {
      steps.unshift({ type: "member", name: cursor.kind.name });
      cursor = cursor.kind.base;
    } else if (
      cursor.kind.type === "Index" ||
      cursor.kind.type === "IndexCell"
    ) {
      const isCell = cursor.kind.type === "IndexCell";
      const indices = cursor.kind.indices.map((idx: IRExpr) =>
        cg.genIndexArg(idx)
      );
      steps.unshift({ type: "index", indices, isCell });
      cursor = cursor.kind.base;
    } else {
      break;
    }
  }
  return { steps, root: cursor };
}

/** Check if all steps in a chain are member accesses (no index/cell steps). */
function allMemberSteps(steps: ChainStep[]): boolean {
  return steps.every(s => s.type === "member");
}

/** Extract member names from a chain of all-member steps. */
function memberNames(steps: ChainStep[]): string[] {
  return steps.map(s => (s as Extract<ChainStep, { type: "member" }>).name);
}

/** Emit read-forward code for a chain of steps, returning temp variable names. */
function emitChainReadForward(
  cg: Codegen,
  rootRef: string,
  steps: ChainStep[]
): string[] {
  const temps: string[] = [];
  for (let i = 0; i < steps.length; i++) {
    const tmp = cg.freshTemp();
    temps.push(tmp);
    const parent = i === 0 ? rootRef : temps[i - 1];
    const step = steps[i];
    if (step.type === "member") {
      cg.emit(
        `var ${tmp} = ${`$rt.getMemberOrEmpty(${parent}, ${JSON.stringify(step.name)})`};`
      );
    } else {
      const fn = step.isCell ? "indexCellOrEmpty" : "builtinIndexOrEmpty";
      cg.emit(
        `var ${tmp} = ${`$rt.${fn}(${parent}, [${step.indices.join(", ")}])`};`
      );
    }
  }
  return temps;
}

/** Emit backpropagation code to store changes back up a chain. */
function emitChainBackpropagate(
  cg: Codegen,
  rootRef: string,
  steps: ChainStep[],
  temps: string[]
): void {
  for (let i = temps.length - 1; i >= 1; i--) {
    emitStepStore(cg, temps[i - 1], steps[i], temps[i]);
  }
  emitStepStore(cg, rootRef, steps[0], temps[0]);
}

/** Emit a single store step (member set or index store). */
function emitStepStore(
  cg: Codegen,
  target: string,
  step: ChainStep,
  value: string
): void {
  if (step.type === "member") {
    cg.emit(
      `${target} = ${`$rt.setMemberReturn(${target}, ${JSON.stringify(step.name)}, ${value})`};`
    );
  } else {
    const fn = step.isCell ? "indexCellStore" : "builtinIndexStore";
    cg.emit(
      `${target} = ${`$rt.${fn}(${target}, [${step.indices.join(", ")}], ${value})`};`
    );
  }
}

/** Choose the right empty-value initializer for a null root based on the
 *  first access step (or the final assignment type if there are no steps). */
function nullInitExpr(
  firstStep: ChainStep | undefined,
  finalIsMember: boolean
): string {
  const step = firstStep;
  if (!step) {
    // No intermediate steps — init based on what the final assignment does
    return finalIsMember ? "$rt.emptyStruct()" : "$rt.emptyTensor()";
  }
  if (step.type === "member") return "$rt.emptyStruct()";
  if (step.type === "index" && step.isCell) return "$rt.makeCell([], [0, 0])";
  return "$rt.emptyTensor()";
}

/**
 * General chain assignment: given a Var root and a chain of access steps,
 * read forward through the chain, apply the final assignment, then
 * backpropagate changes to the root.
 *
 * finalAssign is called with the reference to the deepest intermediate
 * (or rootRef if steps is empty) and should emit the assignment code.
 * finalIsMember indicates whether the final assignment is a member set
 * (used to choose the right null initializer when steps is empty).
 */
function genGeneralChainAssign(
  cg: Codegen,
  rootRef: string,
  steps: ChainStep[],
  finalAssign: (leafRef: string) => void,
  finalIsMember: boolean = true
): void {
  const init = nullInitExpr(steps[0], finalIsMember);
  cg.emit(`if (${rootRef} == null) ${rootRef} = ${init};`);
  if (steps.length === 0) {
    finalAssign(rootRef);
  } else {
    const temps = emitChainReadForward(cg, rootRef, steps);
    const leafRef = temps[temps.length - 1];
    finalAssign(leafRef);
    emitChainBackpropagate(cg, rootRef, steps, temps);
  }
}

// ── Assignment handlers ─────────────────────────────────────────────────

function genMemberAssign(
  cg: Codegen,
  lv: Extract<IRLValue, { type: "Member" }>,
  rhs: string
): void {
  const { steps, root } = flattenExprChain(cg, lv.base);

  if (root.kind.type === "Var") {
    const rootRef = cg.varRef(root.kind.variable.id.id);
    const rootType = itemTypeForExprKind(root.kind);

    // Optimized path: pure member chain with known class type → subsasgn
    if (allMemberSteps(steps)) {
      const chain = [...memberNames(steps), lv.name];

      if (
        rootType.kind === "ClassInstance" &&
        hasClassMethod(cg, rootType.className, "subsasgn") &&
        cg.loweringCtx.ownerClassName !== rootType.className
      ) {
        cg.emit(
          `${rootRef} = ${`$rt.subsasgnCall(${rootRef}, ${JSON.stringify(chain)}, ${rhs})`};`
        );
        return;
      }

      if (rootType.kind === "Unknown") {
        cg.emit(`if (${rootRef} == null) ${rootRef} = $rt.emptyStruct();`);
        cg.emit(
          `${rootRef} = ${`$rt.memberChainAssign(${rootRef}, ${JSON.stringify(chain)}, ${rhs})`};`
        );
        return;
      }
    }

    // General case: arbitrary chain of member/index/cell steps ending with .name = rhs
    genGeneralChainAssign(cg, rootRef, steps, leafRef => {
      cg.emit(
        `${leafRef} = ${`$rt.setMemberReturn(${leafRef}, ${JSON.stringify(lv.name)}, ${rhs})`};`
      );
    });
  } else {
    const base = cg.genExpr(lv.base);
    cg.emit(`${`$rt.setMember(${base}, ${JSON.stringify(lv.name)}, ${rhs})`};`);
  }
}

function genMemberDynamicAssign(
  cg: Codegen,
  lv: Extract<IRLValue, { type: "MemberDynamic" }>,
  rhs: string
): void {
  const nameExpr = cg.genExpr(lv.nameExpr);
  const { steps, root } = flattenExprChain(cg, lv.base);

  if (root.kind.type === "Var") {
    const rootRef = cg.varRef(root.kind.variable.id.id);
    genGeneralChainAssign(cg, rootRef, steps, leafRef => {
      cg.emit(
        `${leafRef} = ${`$rt.setMemberDynamicReturn(${leafRef}, ${nameExpr}, ${rhs})`};`
      );
    });
  } else {
    const base = cg.genExpr(lv.base);
    cg.emit(`${`$rt.setMemberDynamicReturn(${base}, ${nameExpr}, ${rhs})`};`);
  }
}

function genIndexAssign(
  cg: Codegen,
  lv: Extract<IRLValue, { type: "Index" | "IndexCell" }>,
  rhs: string
): void {
  const storeFn = lv.type === "Index" ? "indexStore" : "indexCellStore";
  const indices = lv.indices.map(i => cg.genIndexArg(i));

  if (lv.base.kind.type === "Var") {
    const base = cg.genExpr(lv.base);
    // Inside a class method, obj(k) = same-class-val bypasses subsasgn
    const rootType = itemTypeForExprKind(lv.base.kind);
    const ownerClassName = cg.loweringCtx.ownerClassName;
    const skipSubsasgn =
      storeFn === "indexStore" &&
      rootType.kind === "ClassInstance" &&
      ownerClassName === rootType.className;
    // For Unknown-type base inside a class method, emit a runtime check:
    // bypass subsasgn if the object turns out to be an instance of the
    // current class (class methods access their own instances directly
    // without going through overloaded subsasgn).
    const skipSubsasgnRuntimeCheck =
      storeFn === "indexStore" &&
      !skipSubsasgn &&
      ownerClassName !== null &&
      rootType.kind === "Unknown";
    const skipArg = skipSubsasgn
      ? ", true"
      : skipSubsasgnRuntimeCheck
        ? `, $rt.isClassInstance(${base}, ${JSON.stringify(ownerClassName)})`
        : "";
    cg.emit(
      `${cg.varRef(lv.base.kind.variable.id.id)} = ${`$rt.${storeFn}(${base}, [${indices.join(", ")}], ${rhs}${skipArg})`};`
    );
  } else {
    // General case: flatten the base expression chain
    const { steps, root } = flattenExprChain(cg, lv.base);
    if (root.kind.type === "Var") {
      const rootRef = cg.varRef(root.kind.variable.id.id);
      genGeneralChainAssign(
        cg,
        rootRef,
        steps,
        leafRef => {
          cg.emit(
            `${leafRef} = ${`$rt.${storeFn}(${leafRef}, [${indices.join(", ")}], ${rhs})`};`
          );
        },
        false
      );
    } else {
      // Non-variable root: best-effort
      const base = cg.genExpr(lv.base);
      cg.emit(`${`$rt.${storeFn}(${base}, [${indices.join(", ")}], ${rhs})`};`);
    }
  }
}
