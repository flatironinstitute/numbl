import { type AbstractSyntaxTree, type Stmt } from "./parser/types.js";

export interface ExternalAccessDirectives {
  /** Variable names declared at file/script scope (outside any function) */
  fileScope: Set<string>;
  /** Function name -> variable names declared within that function */
  functionScope: Map<string, Set<string>>;
}

const DIRECTIVE_RE = /^\s*%\s*external-access:\s*(.+)$/;

interface FunctionSpan {
  name: string;
  start: number;
  end: number;
}

/** Collect all Function statement spans from the AST (recursively). */
function collectFunctionSpans(stmts: Stmt[]): FunctionSpan[] {
  const spans: FunctionSpan[] = [];
  for (const stmt of stmts) {
    if (stmt.type === "Function") {
      spans.push({
        name: stmt.name,
        start: stmt.span.start,
        end: stmt.span.end,
      });
      // Recurse into nested functions
      spans.push(...collectFunctionSpans(stmt.body));
    } else if (stmt.type === "If") {
      spans.push(...collectFunctionSpans(stmt.thenBody));
      for (const b of stmt.elseifBlocks)
        spans.push(...collectFunctionSpans(b.body));
      if (stmt.elseBody) spans.push(...collectFunctionSpans(stmt.elseBody));
    } else if (stmt.type === "For" || stmt.type === "While") {
      spans.push(...collectFunctionSpans(stmt.body));
    } else if (stmt.type === "Switch") {
      for (const c of stmt.cases) spans.push(...collectFunctionSpans(c.body));
      if (stmt.otherwise) spans.push(...collectFunctionSpans(stmt.otherwise));
    } else if (stmt.type === "TryCatch") {
      spans.push(...collectFunctionSpans(stmt.tryBody));
      spans.push(...collectFunctionSpans(stmt.catchBody));
    }
  }
  return spans;
}

/**
 * Extract `% external-access: x y z` directives from source and associate
 * them with the correct scope using AST function spans.
 */
export function extractExternalAccessDirectives(
  source: string,
  ast: AbstractSyntaxTree
): ExternalAccessDirectives {
  const fileScope = new Set<string>();
  const functionScope = new Map<string, Set<string>>();

  // Collect function spans, sorted by start position (innermost-first for nesting)
  const funcSpans = collectFunctionSpans(ast.body);
  // Sort by span length ascending so innermost functions come first
  funcSpans.sort((a, b) => a.end - a.start - (b.end - b.start));

  // Scan source lines for directives
  let offset = 0;
  const lines = source.split("\n");
  for (const line of lines) {
    const match = line.match(DIRECTIVE_RE);
    if (match) {
      const varNames = match[1].trim().split(/\s+/);
      const charPos = offset;

      // Find the innermost enclosing function
      let enclosingFunc: string | null = null;
      for (const fs of funcSpans) {
        if (charPos >= fs.start && charPos < fs.end) {
          enclosingFunc = fs.name;
          break; // innermost first due to sort
        }
      }

      if (enclosingFunc) {
        let set = functionScope.get(enclosingFunc);
        if (!set) {
          set = new Set();
          functionScope.set(enclosingFunc, set);
        }
        for (const v of varNames) set.add(v);
      } else {
        for (const v of varNames) fileScope.add(v);
      }
    }
    offset += line.length + 1; // +1 for newline
  }

  return { fileScope, functionScope };
}
