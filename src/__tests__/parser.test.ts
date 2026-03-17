import { describe, it, expect } from "vitest";
import { parseMFile, SyntaxError } from "../numbl-core/parser/index.js";
import type { Stmt, Expr } from "../numbl-core/parser/index.js";

// Helper to get the first statement
function parseFirst(code: string): Stmt {
  return parseMFile(code).body[0];
}

// Helper to get the expression from an ExprStmt
function parseExpr(code: string): Expr {
  const stmt = parseFirst(code + ";");
  if (stmt.type !== "ExprStmt") throw new Error("expected ExprStmt");
  return stmt.expr;
}

// ── Basic Statements ──────────────────────────────────────────────────

describe("parseMFile - statements", () => {
  it("parses a simple assignment", () => {
    const stmt = parseFirst("x = 1;");
    expect(stmt.type).toBe("Assign");
    if (stmt.type === "Assign") {
      expect(stmt.name).toBe("x");
      expect(stmt.suppressed).toBe(true);
    }
  });

  it("parses assignment without suppression", () => {
    const stmt = parseFirst("x = 1");
    expect(stmt.type).toBe("Assign");
    if (stmt.type === "Assign") {
      expect(stmt.suppressed).toBe(false);
    }
  });

  it("parses an expression statement", () => {
    const stmt = parseFirst("42;");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.suppressed).toBe(true);
    }
  });

  it("parses empty input", () => {
    const ast = parseMFile("");
    expect(ast.body).toHaveLength(0);
  });

  it("parses multiple statements", () => {
    const ast = parseMFile("x = 1;\ny = 2;\nz = 3;");
    expect(ast.body).toHaveLength(3);
  });

  it("skips semicolons and commas between statements", () => {
    const ast = parseMFile("x = 1;;;\ny = 2");
    expect(ast.body).toHaveLength(2);
  });

  it("throws on invalid syntax", () => {
    expect(() => parseMFile("if")).toThrow();
  });
});

// ── Control Flow ──────────────────────────────────────────────────────

describe("parseMFile - control flow", () => {
  it("parses if", () => {
    const stmt = parseFirst("if true\n  x = 1;\nend");
    expect(stmt.type).toBe("If");
    if (stmt.type === "If") {
      expect(stmt.thenBody).toHaveLength(1);
      expect(stmt.elseifBlocks).toHaveLength(0);
      expect(stmt.elseBody).toBeNull();
    }
  });

  it("parses if-else", () => {
    const stmt = parseFirst("if x\n  a=1;\nelse\n  a=2;\nend");
    expect(stmt.type).toBe("If");
    if (stmt.type === "If") {
      expect(stmt.elseBody).not.toBeNull();
      expect(stmt.elseBody!).toHaveLength(1);
    }
  });

  it("parses if-elseif-else", () => {
    const stmt = parseFirst(
      "if x > 0\n  y = 1;\nelseif x == 0\n  y = 0;\nelse\n  y = -1;\nend"
    );
    expect(stmt.type).toBe("If");
    if (stmt.type === "If") {
      expect(stmt.elseifBlocks).toHaveLength(1);
      expect(stmt.elseBody).not.toBeNull();
    }
  });

  it("parses multiple elseif", () => {
    const stmt = parseFirst(
      "if a\n  x=1;\nelseif b\n  x=2;\nelseif c\n  x=3;\nelse\n  x=4;\nend"
    );
    if (stmt.type === "If") {
      expect(stmt.elseifBlocks).toHaveLength(2);
    }
  });

  it("parses for loop", () => {
    const stmt = parseFirst("for i = 1:10\n  x = i;\nend");
    expect(stmt.type).toBe("For");
    if (stmt.type === "For") {
      expect(stmt.varName).toBe("i");
    }
  });

  it("parses for loop with parens", () => {
    const stmt = parseFirst("for (i = 1:10)\n  x = i;\nend");
    expect(stmt.type).toBe("For");
    if (stmt.type === "For") {
      expect(stmt.varName).toBe("i");
    }
  });

  it("parses while loop", () => {
    const stmt = parseFirst("while x > 0\n  x = x - 1;\nend");
    expect(stmt.type).toBe("While");
  });

  it("parses switch statement", () => {
    const stmt = parseFirst(
      "switch x\n  case 1\n    y = 1;\n  case 2\n    y = 2;\n  otherwise\n    y = 0;\nend"
    );
    expect(stmt.type).toBe("Switch");
    if (stmt.type === "Switch") {
      expect(stmt.cases).toHaveLength(2);
      expect(stmt.otherwise).not.toBeNull();
    }
  });

  it("parses switch without otherwise", () => {
    const stmt = parseFirst("switch x\n  case 1\n    y = 1;\nend");
    if (stmt.type === "Switch") {
      expect(stmt.otherwise).toBeNull();
    }
  });

  it("parses try-catch", () => {
    const stmt = parseFirst("try\n  x = 1;\ncatch e\n  x = 0;\nend");
    expect(stmt.type).toBe("TryCatch");
    if (stmt.type === "TryCatch") {
      expect(stmt.catchVar).toBe("e");
      expect(stmt.tryBody).toHaveLength(1);
      expect(stmt.catchBody).toHaveLength(1);
    }
  });

  it("parses try-catch without catch variable", () => {
    const stmt = parseFirst("try\n  x = 1;\ncatch\n  x = 0;\nend");
    if (stmt.type === "TryCatch") {
      expect(stmt.catchVar).toBeNull();
    }
  });

  it("parses try without catch", () => {
    const stmt = parseFirst("try\n  x = 1;\nend");
    if (stmt.type === "TryCatch") {
      expect(stmt.catchBody).toHaveLength(0);
    }
  });

  it("parses break", () => {
    const ast = parseMFile("for i=1:10\n  break;\nend");
    const forStmt = ast.body[0];
    if (forStmt.type === "For") {
      expect(forStmt.body.some(s => s.type === "Break")).toBe(true);
    }
  });

  it("parses continue", () => {
    const ast = parseMFile("for i=1:10\n  continue;\nend");
    const forStmt = ast.body[0];
    if (forStmt.type === "For") {
      expect(forStmt.body.some(s => s.type === "Continue")).toBe(true);
    }
  });

  it("parses return", () => {
    const ast = parseMFile("function foo()\n  return;\nend");
    if (ast.body[0].type === "Function") {
      expect(ast.body[0].body.some(s => s.type === "Return")).toBe(true);
    }
  });
});

// ── Functions ─────────────────────────────────────────────────────────

describe("parseMFile - functions", () => {
  it("parses function with single output", () => {
    const stmt = parseFirst("function y = foo(x)\n  y = x + 1;\nend");
    expect(stmt.type).toBe("Function");
    if (stmt.type === "Function") {
      expect(stmt.name).toBe("foo");
      expect(stmt.params).toEqual(["x"]);
      expect(stmt.outputs).toEqual(["y"]);
    }
  });

  it("parses function with multiple outputs", () => {
    const stmt = parseFirst(
      "function [a, b] = foo(x)\n  a = x;\n  b = x + 1;\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.outputs).toEqual(["a", "b"]);
    }
  });

  it("parses function with no outputs", () => {
    const stmt = parseFirst("function foo(x)\n  disp(x);\nend");
    if (stmt.type === "Function") {
      expect(stmt.outputs).toEqual([]);
      expect(stmt.params).toEqual(["x"]);
    }
  });

  it("parses function with no params", () => {
    const stmt = parseFirst("function y = foo()\n  y = 1;\nend");
    if (stmt.type === "Function") {
      expect(stmt.params).toEqual([]);
    }
  });

  it("parses function with no parens", () => {
    const stmt = parseFirst("function foo\n  x = 1;\nend");
    if (stmt.type === "Function") {
      expect(stmt.params).toEqual([]);
      expect(stmt.outputs).toEqual([]);
    }
  });

  it("parses function with tilde output", () => {
    const stmt = parseFirst("function [~, b] = foo(x)\n  b = x;\nend");
    if (stmt.type === "Function") {
      expect(stmt.outputs).toEqual(["~", "b"]);
    }
  });

  it("parses function with varargin", () => {
    const stmt = parseFirst("function foo(a, varargin)\n  x = 1;\nend");
    if (stmt.type === "Function") {
      expect(stmt.params).toEqual(["a", "varargin"]);
    }
  });

  it("parses function with varargout", () => {
    const stmt = parseFirst("function [a, varargout] = foo(x)\n  a = x;\nend");
    if (stmt.type === "Function") {
      expect(stmt.outputs).toEqual(["a", "varargout"]);
    }
  });

  it("parses function without end (auto-insertion)", () => {
    const ast = parseMFile("function y = foo(x)\n  y = x + 1;\n");
    expect(ast.body[0].type).toBe("Function");
  });

  it("parses multiple functions without end", () => {
    const ast = parseMFile(
      "function y = foo(x)\n  y = x + 1;\n\nfunction z = bar(x)\n  z = x * 2;\n"
    );
    expect(ast.body).toHaveLength(2);
    expect(ast.body[0].type).toBe("Function");
    expect(ast.body[1].type).toBe("Function");
  });
});

// ── Global/Persistent ─────────────────────────────────────────────────

describe("parseMFile - global/persistent", () => {
  it("parses global statement", () => {
    const stmt = parseFirst("global x y z");
    expect(stmt.type).toBe("Global");
    if (stmt.type === "Global") {
      expect(stmt.names).toEqual(["x", "y", "z"]);
    }
  });

  it("parses persistent statement", () => {
    const stmt = parseFirst("persistent x y");
    expect(stmt.type).toBe("Persistent");
    if (stmt.type === "Persistent") {
      expect(stmt.names).toEqual(["x", "y"]);
    }
  });

  it("parses global with commas", () => {
    const stmt = parseFirst("global x, y");
    expect(stmt.type).toBe("Global");
  });
});

// ── Expressions ───────────────────────────────────────────────────────

describe("parseMFile - expressions", () => {
  it("parses number literal", () => {
    const expr = parseExpr("42");
    expect(expr.type).toBe("Number");
    if (expr.type === "Number") expect(expr.value).toBe("42");
  });

  it("parses float literal", () => {
    const expr = parseExpr("3.14");
    expect(expr.type).toBe("Number");
  });

  it("parses identifier", () => {
    const expr = parseExpr("myVar");
    expect(expr.type).toBe("Ident");
    if (expr.type === "Ident") expect(expr.name).toBe("myVar");
  });

  it("parses char literal", () => {
    const expr = parseExpr("'hello'");
    expect(expr.type).toBe("Char");
  });

  it("parses string literal", () => {
    const expr = parseExpr('"hello"');
    expect(expr.type).toBe("String");
  });

  it("parses binary addition", () => {
    const expr = parseExpr("a + b");
    expect(expr.type).toBe("Binary");
    if (expr.type === "Binary") expect(expr.op).toBe("Add");
  });

  it("parses binary subtraction", () => {
    const expr = parseExpr("a - b");
    if (expr.type === "Binary") expect(expr.op).toBe("Sub");
  });

  it("parses multiplication", () => {
    const expr = parseExpr("a * b");
    if (expr.type === "Binary") expect(expr.op).toBe("Mul");
  });

  it("parses element-wise multiplication", () => {
    const expr = parseExpr("a .* b");
    if (expr.type === "Binary") expect(expr.op).toBe("ElemMul");
  });

  it("parses division", () => {
    const expr = parseExpr("a / b");
    if (expr.type === "Binary") expect(expr.op).toBe("Div");
  });

  it("parses element-wise division", () => {
    const expr = parseExpr("a ./ b");
    if (expr.type === "Binary") expect(expr.op).toBe("ElemDiv");
  });

  it("parses left division (backslash)", () => {
    const expr = parseExpr("A \\ b");
    if (expr.type === "Binary") expect(expr.op).toBe("LeftDiv");
  });

  it("parses element-wise left division", () => {
    const expr = parseExpr("A .\\ b");
    if (expr.type === "Binary") expect(expr.op).toBe("ElemLeftDiv");
  });

  it("parses power", () => {
    const expr = parseExpr("a ^ b");
    if (expr.type === "Binary") expect(expr.op).toBe("Pow");
  });

  it("parses element-wise power", () => {
    const expr = parseExpr("a .^ b");
    if (expr.type === "Binary") expect(expr.op).toBe("ElemPow");
  });

  it("parses comparison operators", () => {
    for (const [code, op] of [
      ["a == b", "Equal"],
      ["a ~= b", "NotEqual"],
      ["a < b", "Less"],
      ["a <= b", "LessEqual"],
      ["a > b", "Greater"],
      ["a >= b", "GreaterEqual"],
    ] as const) {
      const expr = parseExpr(code);
      expect(expr.type).toBe("Binary");
      if (expr.type === "Binary") expect(expr.op).toBe(op);
    }
  });

  it("parses logical operators", () => {
    const expr = parseExpr("a && b || c");
    expect(expr.type).toBe("Binary");
    if (expr.type === "Binary") expect(expr.op).toBe("OrOr");
  });

  it("parses bitwise operators", () => {
    const exprAnd = parseExpr("a & b");
    if (exprAnd.type === "Binary") expect(exprAnd.op).toBe("BitAnd");
    const exprOr = parseExpr("a | b");
    if (exprOr.type === "Binary") expect(exprOr.op).toBe("BitOr");
  });

  it("parses unary minus", () => {
    const expr = parseExpr("-x");
    expect(expr.type).toBe("Unary");
    if (expr.type === "Unary") expect(expr.op).toBe("Minus");
  });

  it("parses unary plus", () => {
    const expr = parseExpr("+x");
    expect(expr.type).toBe("Unary");
    if (expr.type === "Unary") expect(expr.op).toBe("Plus");
  });

  it("parses logical not", () => {
    const expr = parseExpr("~x");
    expect(expr.type).toBe("Unary");
    if (expr.type === "Unary") expect(expr.op).toBe("Not");
  });

  it("parses transpose", () => {
    const expr = parseExpr("x'");
    expect(expr.type).toBe("Unary");
    if (expr.type === "Unary") expect(expr.op).toBe("Transpose");
  });

  it("parses non-conjugate transpose (.')", () => {
    const expr = parseExpr("x.'");
    expect(expr.type).toBe("Unary");
    if (expr.type === "Unary") expect(expr.op).toBe("NonConjugateTranspose");
  });

  it("parses range expression", () => {
    const expr = parseExpr("1:10");
    expect(expr.type).toBe("Range");
    if (expr.type === "Range") {
      expect(expr.step).toBeNull();
    }
  });

  it("parses range with step", () => {
    const expr = parseExpr("1:2:10");
    expect(expr.type).toBe("Range");
    if (expr.type === "Range") {
      expect(expr.step).not.toBeNull();
    }
  });

  it("parses function call", () => {
    const expr = parseExpr("foo(1, 2)");
    expect(expr.type).toBe("FuncCall");
    if (expr.type === "FuncCall") {
      expect(expr.name).toBe("foo");
      expect(expr.args).toHaveLength(2);
    }
  });

  it("parses function call with no args", () => {
    const expr = parseExpr("foo()");
    if (expr.type === "FuncCall") {
      expect(expr.args).toHaveLength(0);
    }
  });

  it("parses cell indexing with braces", () => {
    const expr = parseExpr("c{1}");
    expect(expr.type).toBe("IndexCell");
  });

  it("parses member access", () => {
    const expr = parseExpr("obj.field");
    expect(expr.type).toBe("Member");
    if (expr.type === "Member") {
      expect(expr.name).toBe("field");
    }
  });

  it("parses method call", () => {
    const expr = parseExpr("obj.method(1)");
    expect(expr.type).toBe("MethodCall");
    if (expr.type === "MethodCall") {
      expect(expr.name).toBe("method");
    }
  });

  it("parses dynamic field access", () => {
    const expr = parseExpr("s.(name)");
    expect(expr.type).toBe("MemberDynamic");
  });

  it("parses anonymous function", () => {
    const stmt = parseFirst("f = @(x) x + 1;");
    expect(stmt.type).toBe("Assign");
    if (stmt.type === "Assign") {
      expect(stmt.expr.type).toBe("AnonFunc");
      if (stmt.expr.type === "AnonFunc") {
        expect(stmt.expr.params).toEqual(["x"]);
      }
    }
  });

  it("parses anonymous function with no params", () => {
    const stmt = parseFirst("f = @() 42;");
    if (stmt.type === "Assign" && stmt.expr.type === "AnonFunc") {
      expect(stmt.expr.params).toEqual([]);
    }
  });

  it("parses function handle", () => {
    const expr = parseExpr("@sin");
    expect(expr.type).toBe("FuncHandle");
    if (expr.type === "FuncHandle") {
      expect(expr.name).toBe("sin");
    }
  });

  it("parses matrix literal (Tensor)", () => {
    const expr = parseExpr("[1, 2; 3, 4]");
    expect(expr.type).toBe("Tensor");
    if (expr.type === "Tensor") {
      expect(expr.rows).toHaveLength(2);
      expect(expr.rows[0]).toHaveLength(2);
    }
  });

  it("parses empty matrix", () => {
    const expr = parseExpr("[]");
    expect(expr.type).toBe("Tensor");
    if (expr.type === "Tensor") {
      expect(expr.rows).toHaveLength(0);
    }
  });

  it("parses row vector", () => {
    const expr = parseExpr("[1, 2, 3]");
    if (expr.type === "Tensor") {
      expect(expr.rows).toHaveLength(1);
      expect(expr.rows[0]).toHaveLength(3);
    }
  });

  it("parses cell literal", () => {
    const expr = parseExpr("{1, 'hello'}");
    expect(expr.type).toBe("Cell");
  });

  it("parses empty cell", () => {
    const expr = parseExpr("{}");
    expect(expr.type).toBe("Cell");
    if (expr.type === "Cell") {
      expect(expr.rows).toHaveLength(0);
    }
  });

  it("parses cell with rows", () => {
    const expr = parseExpr("{1, 2; 3, 4}");
    if (expr.type === "Cell") {
      expect(expr.rows).toHaveLength(2);
    }
  });

  it("parses parenthesized expression", () => {
    const expr = parseExpr("(1 + 2)");
    expect(expr.type).toBe("Binary");
  });

  it("parses colon expression", () => {
    const expr = parseExpr(":");
    expect(expr.type).toBe("Colon");
  });

  it("parses end keyword in index", () => {
    const expr = parseExpr("a(end)");
    expect(expr.type).toBe("FuncCall");
  });

  it("parses true and false", () => {
    const t = parseExpr("true");
    expect(t.type).toBe("Ident");
    if (t.type === "Ident") expect(t.name).toBe("true");

    const f = parseExpr("false");
    expect(f.type).toBe("Ident");
    if (f.type === "Ident") expect(f.name).toBe("false");
  });

  it("parses imaginary unit", () => {
    const expr = parseExpr("3i");
    expect(expr.type).toBe("Binary");
  });

  it("parses meta-class query", () => {
    const expr = parseExpr("?MyClass");
    expect(expr.type).toBe("MetaClass");
  });

  it("parses power with unary exponent", () => {
    const expr = parseExpr("2^-3");
    expect(expr.type).toBe("Binary");
    if (expr.type === "Binary") {
      expect(expr.op).toBe("Pow");
      expect(expr.right.type).toBe("Unary");
    }
  });

  it("parses chained member access and calls", () => {
    const expr = parseExpr("a.b.c(1)");
    expect(expr.type).toBe("MethodCall");
  });

  it("parses whitespace-separated matrix elements", () => {
    const expr = parseExpr("[1 2 3]");
    if (expr.type === "Tensor") {
      expect(expr.rows[0]).toHaveLength(3);
    }
  });

  it("parses matrix with newline as row separator", () => {
    const expr = parseExpr("[1 2\n3 4]");
    if (expr.type === "Tensor") {
      expect(expr.rows).toHaveLength(2);
    }
  });
});

// ── LValue Assignment ─────────────────────────────────────────────────

describe("parseMFile - lvalue assignment", () => {
  it("parses indexed assignment", () => {
    const stmt = parseFirst("a(1) = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("Index");
    }
  });

  it("parses cell assignment", () => {
    const stmt = parseFirst("c{1} = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("IndexCell");
    }
  });

  it("parses field assignment", () => {
    const stmt = parseFirst("s.x = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("Member");
    }
  });

  it("parses dynamic field assignment", () => {
    const stmt = parseFirst("s.(name) = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("MemberDynamic");
    }
  });

  it("parses chained index assignment", () => {
    const stmt = parseFirst("a(1).b = 5;");
    expect(stmt.type).toBe("AssignLValue");
  });
});

// ── Multi-assign ──────────────────────────────────────────────────────

describe("parseMFile - multi-assign", () => {
  it("parses multiple assignments", () => {
    const stmt = parseFirst("[a, b] = foo();");
    expect(stmt.type).toBe("MultiAssign");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues).toHaveLength(2);
    }
  });

  it("parses multi-assign with tilde (ignore)", () => {
    const stmt = parseFirst("[~, b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("Ignore");
      expect(stmt.lvalues[1].type).toBe("Var");
    }
  });

  it("falls back to expression when not multi-assign", () => {
    const stmt = parseFirst("[1, 2, 3];");
    expect(stmt.type).toBe("ExprStmt");
  });

  it("parses multi-assign with cell indexing", () => {
    const stmt = parseFirst("[a{1}, b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("IndexCell");
    }
  });
});

// ── Command Form ──────────────────────────────────────────────────────

describe("parseMFile - command form", () => {
  it("parses hold on as command", () => {
    const stmt = parseFirst("hold on");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.expr.type).toBe("FuncCall");
      if (stmt.expr.type === "FuncCall") {
        expect(stmt.expr.name).toBe("hold");
      }
    }
  });

  it("parses grid off as command", () => {
    const stmt = parseFirst("grid off");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("grid");
    }
  });

  it("parses clear as command", () => {
    const stmt = parseFirst("clear x y z");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("clear");
      expect(stmt.expr.args).toHaveLength(3);
    }
  });

  it("parses pause without args", () => {
    const stmt = parseFirst("pause\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("pause");
    }
  });
});

// ── Import ────────────────────────────────────────────────────────────

describe("parseMFile - import", () => {
  it("parses import statement", () => {
    const stmt = parseFirst("import pkg.Class");
    expect(stmt.type).toBe("Import");
    if (stmt.type === "Import") {
      expect(stmt.path).toEqual(["pkg", "Class"]);
      expect(stmt.wildcard).toBe(false);
    }
  });

  it("parses wildcard import", () => {
    const stmt = parseFirst("import pkg.*");
    if (stmt.type === "Import") {
      expect(stmt.wildcard).toBe(true);
    }
  });

  it("parses import() as function call", () => {
    const stmt = parseFirst("import('data.mat');");
    expect(stmt.type).toBe("ExprStmt");
  });
});

// ── ClassDef ──────────────────────────────────────────────────────────

describe("parseMFile - classdef", () => {
  it("parses simple classdef", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  properties\n    x\n  end\nend"
    );
    expect(stmt.type).toBe("ClassDef");
    if (stmt.type === "ClassDef") {
      expect(stmt.name).toBe("MyClass");
      expect(stmt.superClass).toBeNull();
      expect(stmt.members).toHaveLength(1);
      expect(stmt.members[0].type).toBe("Properties");
    }
  });

  it("parses classdef with superclass", () => {
    const stmt = parseFirst(
      "classdef MyClass < BaseClass\n  properties\n    x\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      expect(stmt.superClass).toBe("BaseClass");
    }
  });

  it("parses classdef with methods", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods\n    function foo(obj)\n    end\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      expect(methods).toBeDefined();
    }
  });

  it("parses classdef with properties and default values", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  properties\n    x = 0\n    y = 'hello'\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const props = stmt.members[0];
      if (props.type === "Properties") {
        expect(props.names).toEqual(["x", "y"]);
        expect(props.defaultValues[0]).not.toBeNull();
        expect(props.defaultValues[1]).not.toBeNull();
      }
    }
  });

  it("parses classdef with attributes", () => {
    const stmt = parseFirst(
      "classdef (Sealed) MyClass\n  properties (Access = private)\n    x\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      expect(stmt.classAttributes).toHaveLength(1);
      expect(stmt.classAttributes[0].name).toBe("Sealed");
    }
  });

  it("parses classdef with events", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  events\n    MyEvent\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const events = stmt.members.find(m => m.type === "Events");
      expect(events).toBeDefined();
    }
  });

  it("parses classdef with enumeration", () => {
    const stmt = parseFirst(
      "classdef MyEnum\n  enumeration\n    Red\n    Green\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const enums = stmt.members.find(m => m.type === "Enumeration");
      expect(enums).toBeDefined();
    }
  });
});

// ── SyntaxError ───────────────────────────────────────────────────────

describe("SyntaxError", () => {
  it("has correct properties", () => {
    const err = new SyntaxError("test error", 42, "foo", "'bar'", 5);
    expect(err.message).toBe("test error");
    expect(err.position).toBe(42);
    expect(err.foundToken).toBe("foo");
    expect(err.expected).toBe("'bar'");
    expect(err.line).toBe(5);
    expect(err.name).toBe("SyntaxError");
  });

  it("toString with line", () => {
    const err = new SyntaxError("bad token", 0, "x", null, 3);
    const str = err.toString();
    expect(str).toContain("line 3");
    expect(str).toContain("bad token");
    expect(str).toContain("found: 'x'");
  });

  it("toString with file and line", () => {
    const err = new SyntaxError("bad token", 0, null, null, 3);
    err.file = "test.m";
    const str = err.toString();
    expect(str).toContain("test.m");
    expect(str).toContain("line 3");
  });

  it("toString with expected", () => {
    const err = new SyntaxError("unexpected", 0, null, "'end'", null);
    const str = err.toString();
    expect(str).toContain("expected: 'end'");
  });

  it("toString without line", () => {
    const err = new SyntaxError("bad", 10, null, null, null);
    const str = err.toString();
    expect(str).toContain("position 10");
  });
});

// ── Arguments blocks ──────────────────────────────────────────────────

describe("parseMFile - arguments blocks", () => {
  it("parses function with arguments block", () => {
    const stmt = parseFirst(
      "function foo(x, y)\n  arguments\n    x\n    y\n  end\n  z = x + y;\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks).toHaveLength(1);
      expect(stmt.argumentsBlocks[0].kind).toBe("Input");
      expect(stmt.argumentsBlocks[0].entries).toHaveLength(2);
    }
  });

  it("parses arguments block with default values", () => {
    const stmt = parseFirst(
      "function foo(x, y)\n  arguments\n    x = 1\n    y = 2\n  end\n  z = x;\nend"
    );
    if (stmt.type === "Function") {
      const block = stmt.argumentsBlocks[0];
      expect(block.entries[0].defaultValue).not.toBeNull();
    }
  });

  it("parses arguments block with class name", () => {
    const stmt = parseFirst(
      "function foo(x)\n  arguments\n    x double\n  end\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks[0].entries[0].className).toBe("double");
    }
  });

  it("parses Output arguments block", () => {
    const stmt = parseFirst(
      "function y = foo(x)\n  arguments (Output)\n    y\n  end\n  y = x;\nend"
    );
    if (stmt.type === "Function") {
      const block = stmt.argumentsBlocks.find(b => b.kind === "Output");
      expect(block).toBeDefined();
    }
  });

  it("parses Repeating arguments block", () => {
    const stmt = parseFirst(
      "function foo(varargin)\n  arguments (Repeating)\n    x\n  end\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks[0].kind).toBe("Repeating");
    }
  });

  it("parses Output,Repeating arguments block", () => {
    const stmt = parseFirst(
      "function varargout = foo(x)\n  arguments (Output, Repeating)\n    y\n  end\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks[0].kind).toBe("OutputRepeating");
    }
  });

  it("parses struct.field argument name", () => {
    const stmt = parseFirst(
      "function foo(opts)\n  arguments\n    opts.Name\n  end\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks[0].entries[0].name).toBe("opts.Name");
    }
  });

  it("parses argument with dimensions", () => {
    const stmt = parseFirst(
      "function foo(x)\n  arguments\n    x (1,:) double\n  end\nend"
    );
    if (stmt.type === "Function") {
      const entry = stmt.argumentsBlocks[0].entries[0];
      expect(entry.dimensions).toEqual(["1", ":"]);
      expect(entry.className).toBe("double");
    }
  });

  it("parses argument with named dimensions", () => {
    const stmt = parseFirst(
      "function foo(x)\n  arguments\n    x (m,n)\n  end\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks[0].entries[0].dimensions).toEqual(["m", "n"]);
    }
  });

  it("parses argument with validators", () => {
    const stmt = parseFirst(
      "function foo(x)\n  arguments\n    x {mustBeNumeric, mustBePositive}\n  end\nend"
    );
    if (stmt.type === "Function") {
      const entry = stmt.argumentsBlocks[0].entries[0];
      expect(entry.validators).toEqual(["mustBeNumeric", "mustBePositive"]);
    }
  });

  it("parses argument with dimensions, class, validators, and default", () => {
    const stmt = parseFirst(
      "function foo(x)\n  arguments\n    x (1,:) double {mustBePositive} = 1\n  end\nend"
    );
    if (stmt.type === "Function") {
      const entry = stmt.argumentsBlocks[0].entries[0];
      expect(entry.dimensions).toEqual(["1", ":"]);
      expect(entry.className).toBe("double");
      expect(entry.validators).toEqual(["mustBePositive"]);
      expect(entry.defaultValue).not.toBeNull();
    }
  });

  it("parses multiple arguments blocks", () => {
    const stmt = parseFirst(
      "function y = foo(x)\n  arguments\n    x\n  end\n  arguments (Output)\n    y\n  end\n  y = x;\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.argumentsBlocks).toHaveLength(2);
      expect(stmt.argumentsBlocks[0].kind).toBe("Input");
      expect(stmt.argumentsBlocks[1].kind).toBe("Output");
    }
  });
});

// ── Additional ClassDef coverage ─────────────────────────────────────

describe("parseMFile - classdef advanced", () => {
  it("parses abstract methods block with signatures", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods (Abstract)\n    result = doSomething(obj, x)\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      expect(methods).toBeDefined();
      if (methods?.type === "Methods") {
        expect(methods.body).toHaveLength(0);
        expect(methods.signatures).toBeDefined();
        expect(methods.signatures![0].name).toBe("doSomething");
        expect(methods.signatures![0].outputs).toEqual(["result"]);
        expect(methods.signatures![0].params).toEqual(["obj", "x"]);
      }
    }
  });

  it("parses abstract method signature without outputs", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods (Abstract)\n    doIt(obj)\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      if (methods?.type === "Methods") {
        expect(methods.signatures![0].name).toBe("doIt");
        expect(methods.signatures![0].outputs).toEqual([]);
      }
    }
  });

  it("parses abstract method signature with multiple outputs", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods (Abstract)\n    [a, b] = compute(obj)\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      if (methods?.type === "Methods") {
        expect(methods.signatures![0].outputs).toEqual(["a", "b"]);
      }
    }
  });

  it("parses method signatures in non-abstract methods block", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods\n    function foo(obj)\n    end\n    bar(obj)\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      if (methods?.type === "Methods") {
        expect(methods.body).toHaveLength(1);
        expect(methods.signatures).toBeDefined();
        expect(methods.signatures![0].name).toBe("bar");
      }
    }
  });

  it("parses property accessor method (get./set.)", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods\n    function val = get.Name(obj)\n      val = obj.Name;\n    end\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      if (methods?.type === "Methods") {
        const fn = methods.body[0];
        if (fn.type === "Function") {
          expect(fn.name).toBe("get.Name");
        }
      }
    }
  });

  it("parses classdef with attributes including brace-delimited value", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  properties (Access = private, SetAccess = {?ClassA, ?ClassB})\n    x\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const props = stmt.members[0];
      if (props.type === "Properties") {
        expect(props.attributes).toHaveLength(2);
        expect(props.attributes[0].name).toBe("Access");
        expect(props.attributes[0].value).toBe("private");
        expect(props.attributes[1].name).toBe("SetAccess");
        expect(props.attributes[1].value).toContain("?ClassA");
      }
    }
  });

  it("parses classdef with arguments block member", () => {
    const stmt = parseFirst("classdef MyClass\n  arguments\n    x\n  end\nend");
    if (stmt.type === "ClassDef") {
      const args = stmt.members.find(m => m.type === "Arguments");
      expect(args).toBeDefined();
    }
  });

  it("parses classdef with qualified superclass name", () => {
    const stmt = parseFirst(
      "classdef MyClass < pkg.BaseClass\n  properties\n    x\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      expect(stmt.superClass).toBe("pkg.BaseClass");
    }
  });

  it("parses abstract method with get./set. accessor signature", () => {
    const stmt = parseFirst(
      "classdef MyClass\n  methods (Abstract)\n    get.Prop(obj)\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const methods = stmt.members.find(m => m.type === "Methods");
      if (methods?.type === "Methods") {
        expect(methods.signatures![0].name).toBe("get.Prop");
      }
    }
  });
});

// ── Additional Command Form coverage ─────────────────────────────────

describe("parseMFile - command form advanced", () => {
  it("parses command with char argument", () => {
    const stmt = parseFirst("hold 'on'");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("hold");
      expect(stmt.expr.args).toHaveLength(1);
    }
  });

  it("parses command with string argument", () => {
    const stmt = parseFirst('hold "on"');
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("hold");
      expect(stmt.expr.args).toHaveLength(1);
    }
  });

  it("parses command with numeric argument", () => {
    const stmt = parseFirst("clear 42");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("clear");
    }
  });

  it("rejects keyword command with unsupported keyword", () => {
    expect(() => parseMFile("hold badarg\n")).toThrow(/does not support/);
  });

  it("rejects keyword command with multiple args", () => {
    expect(() => parseMFile("hold on off\n")).toThrow(/only one argument/);
  });

  it("parses colormap verb with expr argument", () => {
    const stmt = parseFirst("colormap hsv(600)\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("colormap");
      expect(stmt.expr.args).toHaveLength(1);
    }
  });

  it("parses warning off as command", () => {
    const stmt = parseFirst("warning off\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("warning");
    }
  });

  it("parses uiwait without args (optional keyword)", () => {
    const stmt = parseFirst("uiwait\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("uiwait");
      expect(stmt.expr.args).toHaveLength(0);
    }
  });
});

// ── Additional Expression coverage ───────────────────────────────────

describe("parseMFile - expressions advanced", () => {
  it("parses function handle with dotted name", () => {
    const expr = parseExpr("@pkg.func");
    expect(expr.type).toBe("FuncHandle");
    if (expr.type === "FuncHandle") {
      expect(expr.name).toBe("pkg.func");
    }
  });

  it("parses super method call", () => {
    const expr = parseExpr("foo@Base(x)");
    expect(expr.type).toBe("SuperMethodCall");
    if (expr.type === "SuperMethodCall") {
      expect(expr.methodName).toBe("foo");
      expect(expr.superClassName).toBe("Base");
    }
  });

  it("parses super method call with multiple args", () => {
    const expr = parseExpr("init@Base(a, b)");
    if (expr.type === "SuperMethodCall") {
      expect(expr.args).toHaveLength(2);
    }
  });

  it("parses super method call with no args", () => {
    const expr = parseExpr("init@Base()");
    if (expr.type === "SuperMethodCall") {
      expect(expr.args).toHaveLength(0);
    }
  });

  it("parses indexing on non-ident base", () => {
    const expr = parseExpr("foo()(1)");
    expect(expr.type).toBe("Index");
    if (expr.type === "Index") {
      expect(expr.base.type).toBe("FuncCall");
    }
  });

  it("parses bracket indexing as postfix", () => {
    const stmt = parseFirst("x = a(1);\ny = x[1];");
    // bracket indexing after non-matrix context
    expect(stmt.type).toBe("Assign");
  });

  it("parses .+ operator (element-wise add)", () => {
    const expr = parseExpr("a .+ b");
    expect(expr.type).toBe("Binary");
    if (expr.type === "Binary") {
      expect(expr.op).toBe("Add");
    }
  });

  it("parses .- operator (element-wise sub)", () => {
    const expr = parseExpr("a .- b");
    expect(expr.type).toBe("Binary");
    if (expr.type === "Binary") {
      expect(expr.op).toBe("Sub");
    }
  });

  it("parses cell literal with whitespace-separated elements", () => {
    const expr = parseExpr("{1 2 3}");
    expect(expr.type).toBe("Cell");
    if (expr.type === "Cell") {
      expect(expr.rows[0]).toHaveLength(3);
    }
  });

  it("parses cell with newline row separator", () => {
    const expr = parseExpr("{1 2\n3 4}");
    if (expr.type === "Cell") {
      expect(expr.rows).toHaveLength(2);
    }
  });

  it("parses meta-class with dotted path", () => {
    const expr = parseExpr("?pkg.MyClass");
    expect(expr.type).toBe("MetaClass");
    if (expr.type === "MetaClass") {
      expect(expr.name).toBe("pkg.MyClass");
    }
  });

  it("parses member access with keyword name", () => {
    // After '.', keywords should be valid member names
    const expr = parseExpr("s.for");
    expect(expr.type).toBe("Member");
    if (expr.type === "Member") {
      expect(expr.name).toBe("for");
    }
  });

  it("parses method call with keyword member name", () => {
    const expr = parseExpr("s.end(1)");
    expect(expr.type).toBe("MethodCall");
    if (expr.type === "MethodCall") {
      expect(expr.name).toBe("end");
    }
  });

  it("throws on unexpected end of input", () => {
    expect(() => parseMFile("x = ;")).toThrow();
  });

  it("throws on unexpected token in primary", () => {
    expect(() => parseMFile("x = *;")).toThrow(/unexpected token/);
  });

  it("parses imaginary unit with float", () => {
    const expr = parseExpr("1.5j");
    expect(expr.type).toBe("Binary");
  });

  it("parses matrix with trailing comma", () => {
    const expr = parseExpr("[1, 2,; 3, 4]");
    if (expr.type === "Tensor") {
      expect(expr.rows).toHaveLength(2);
    }
  });

  it("parses anonymous function with multiple params", () => {
    const expr = parseExpr("@(x,y,z) x+y+z");
    expect(expr.type).toBe("AnonFunc");
    if (expr.type === "AnonFunc") {
      expect(expr.params).toEqual(["x", "y", "z"]);
    }
  });

  it("parses cell indexing with multiple indices", () => {
    const expr = parseExpr("c{1,2}");
    expect(expr.type).toBe("IndexCell");
  });

  it("parses method call with multiple args", () => {
    const expr = parseExpr("obj.method(1, 2, 3)");
    if (expr.type === "MethodCall") {
      expect(expr.args).toHaveLength(3);
    }
  });

  it("parses chained postfix operations", () => {
    const expr = parseExpr("a(1).b{2}");
    expect(expr.type).toBe("IndexCell");
  });
});

// ── Additional Function coverage ─────────────────────────────────────

describe("parseMFile - functions advanced", () => {
  it("throws if varargin is not last param", () => {
    expect(() => parseMFile("function foo(varargin, x)\nend")).toThrow(
      /varargin.*last/
    );
  });

  it("throws if varargout is not last output", () => {
    expect(() => parseMFile("function [varargout, x] = foo()\nend")).toThrow(
      /varargout.*last/
    );
  });

  it("parses function with get. accessor name", () => {
    const stmt = parseFirst(
      "function val = get.Name(obj)\n  val = obj.name;\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.name).toBe("get.Name");
    }
  });

  it("parses function with set. accessor name", () => {
    const stmt = parseFirst(
      "function set.Name(obj, val)\n  obj.name = val;\nend"
    );
    if (stmt.type === "Function") {
      expect(stmt.name).toBe("set.Name");
    }
  });

  it("parses function with tilde param", () => {
    const stmt = parseFirst("function foo(~, x)\n  y = x;\nend");
    if (stmt.type === "Function") {
      expect(stmt.params).toEqual(["~", "x"]);
    }
  });
});

// ── Additional LValue / Multi-assign coverage ────────────────────────

describe("parseMFile - lvalue and multi-assign advanced", () => {
  it("parses bracket-indexed lvalue assignment", () => {
    const stmt = parseFirst("a[1] = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("Index");
    }
  });

  it("parses multi-assign with indexed lvalue", () => {
    const stmt = parseFirst("[a(1), b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("Index");
    }
  });

  it("parses multi-assign with member lvalue", () => {
    const stmt = parseFirst("[a.x, b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("Member");
    }
  });

  it("parses multi-assign with dynamic member lvalue", () => {
    const stmt = parseFirst("[a.(name), b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("MemberDynamic");
    }
  });

  it("parses multi-assign with bracket-indexed lvalue", () => {
    const stmt = parseFirst("[a[1], b] = foo();");
    if (stmt.type === "MultiAssign") {
      expect(stmt.lvalues[0].type).toBe("Index");
    }
  });

  it("parses chained cell+member lvalue assignment", () => {
    const stmt = parseFirst("a{1}.x = 5;");
    expect(stmt.type).toBe("AssignLValue");
  });

  it("parses dynamic field lvalue assignment", () => {
    const stmt = parseFirst("a.(name) = 5;");
    expect(stmt.type).toBe("AssignLValue");
    if (stmt.type === "AssignLValue") {
      expect(stmt.lvalue.type).toBe("MemberDynamic");
    }
  });

  it("falls back when LBracket is not multi-assign (no '=')", () => {
    const stmt = parseFirst("[1 + 2];");
    expect(stmt.type).toBe("ExprStmt");
  });
});

// ── Parser index coverage (error tokens, two-pass, ellipsis) ─────────

describe("parseMFile - index.ts coverage", () => {
  it("throws SyntaxError on invalid token", () => {
    expect(() => parseMFile("x = `bad`;")).toThrow(SyntaxError);
  });

  it("handles function files without end (two-pass)", () => {
    const ast = parseMFile("function y = foo(x)\n  y = x + 1;\n");
    expect(ast.body[0].type).toBe("Function");
  });

  it("parses import with DotStar token", () => {
    const stmt = parseFirst("import pkg.sub.*");
    if (stmt.type === "Import") {
      expect(stmt.wildcard).toBe(true);
      expect(stmt.path).toEqual(["pkg", "sub"]);
    }
  });
});

// ── SyntaxError additional coverage ──────────────────────────────────

describe("SyntaxError - additional coverage", () => {
  it("captures expected in parser errors", () => {
    try {
      parseMFile("function [a = foo()\nend");
    } catch (e) {
      if (e instanceof SyntaxError) {
        expect(e.expected).toBeDefined();
      }
    }
  });

  it("handles errors at end of input", () => {
    try {
      parseMFile("for i =");
    } catch (e) {
      if (e instanceof SyntaxError) {
        expect(e.position).toBeGreaterThanOrEqual(0);
      }
    }
  });
});

// ── Block parsing and suppression ────────────────────────────────────

describe("parseMFile - block parsing", () => {
  it("handles AssignLValue suppression in blocks", () => {
    const ast = parseMFile("for i=1:3\n  a(i) = i;\nend");
    const forStmt = ast.body[0];
    if (forStmt.type === "For") {
      const body0 = forStmt.body[0];
      if (body0.type === "AssignLValue") {
        expect(body0.suppressed).toBe(true);
      }
    }
  });

  it("handles MultiAssign suppression in blocks", () => {
    const ast = parseMFile("for i=1:3\n  [a, b] = foo();\nend");
    const forStmt = ast.body[0];
    if (forStmt.type === "For") {
      const body0 = forStmt.body[0];
      if (body0.type === "MultiAssign") {
        expect(body0.suppressed).toBe(true);
      }
    }
  });

  it("handles LBracket in block (multi-assign)", () => {
    const ast = parseMFile("if true\n  [a, b] = foo();\nend");
    const ifStmt = ast.body[0];
    if (ifStmt.type === "If") {
      expect(ifStmt.thenBody[0].type).toBe("MultiAssign");
    }
  });

  it("switch with comma between cases", () => {
    const stmt = parseFirst("switch x, case 1, y = 1;, case 2, y = 2;, end");
    expect(stmt.type).toBe("Switch");
    if (stmt.type === "Switch") {
      expect(stmt.cases).toHaveLength(2);
    }
  });
});

// ── Error path coverage ──────────────────────────────────────────────

describe("parseMFile - error paths", () => {
  it("throws on missing end for for loop", () => {
    expect(() => parseMFile("for i = 1:10\n  x = i;\n")).toThrow();
  });

  it("throws on missing end for while loop", () => {
    expect(() => parseMFile("while true\n  x = 1;\n")).toThrow();
  });

  it("throws on missing end for switch", () => {
    expect(() => parseMFile("switch x\n  case 1\n    y=1;\n")).toThrow();
  });

  it("throws on missing end for try-catch", () => {
    expect(() => parseMFile("try\n  x = 1;\ncatch\n  y = 1;\n")).toThrow();
  });

  it("throws on missing = in for loop", () => {
    expect(() => parseMFile("for i 1:10\nend")).toThrow(/expected '='/);
  });

  it("throws on missing ) in for loop with parens", () => {
    expect(() => parseMFile("for (i = 1:10\nend")).toThrow(/expected '\)'/);
  });

  it("throws on unclosed paren in call", () => {
    expect(() => parseMFile("foo(1, 2\n")).toThrow(/expected '\)'/);
  });

  it("throws on unclosed bracket in matrix", () => {
    expect(() => parseMFile("[1, 2\n")).toThrow();
  });

  it("throws on unclosed brace in cell", () => {
    expect(() => parseMFile("{1, 2\n")).toThrow();
  });

  it("throws on unclosed brace in cell indexing", () => {
    expect(() => parseMFile("c{1\n")).toThrow(/expected '\}'/);
  });

  it("throws on unclosed bracket in indexing", () => {
    // bracket indexing where ] is missing
    expect(() => parseMFile("a = foo();\na[1\n")).toThrow(/expected '\]'/);
  });

  it("throws on unclosed paren in dynamic field", () => {
    expect(() => parseMFile("s.(name\n")).toThrow(/expected '\)'/);
  });

  it("throws on unclosed paren in method call", () => {
    expect(() => parseMFile("obj.method(1, 2\n")).toThrow(/expected '\)'/);
  });

  it("throws on unclosed paren in anon func params", () => {
    expect(() => parseMFile("f = @(x, y 1;\n")).toThrow(/expected '\)'/);
  });

  it("throws on unclosed paren in primary expr", () => {
    expect(() => parseMFile("x = (1 + 2;\n")).toThrow(/expected '\)'/);
  });

  it("throws on missing ] in function outputs", () => {
    expect(() => parseMFile("function [a, b = foo()\nend")).toThrow(
      /expected '\]'/
    );
  });

  it("throws on missing = after function outputs", () => {
    expect(() => parseMFile("function [a, b] foo()\nend")).toThrow(
      /expected '='/
    );
  });

  it("throws on unclosed paren in function params", () => {
    expect(() => parseMFile("function foo(a, b\nend")).toThrow(/expected '\)'/);
  });

  it("parses adjacency as general command syntax", () => {
    const stmt = parseFirst("noncommand b(1)\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.expr.type).toBe("FuncCall");
      if (stmt.expr.type === "FuncCall") {
        expect(stmt.expr.name).toBe("noncommand");
        expect(stmt.expr.args).toHaveLength(1);
        expect(stmt.expr.args[0].type).toBe("Char");
        if (stmt.expr.args[0].type === "Char") {
          expect(stmt.expr.args[0].value).toBe("'b(1)'");
        }
      }
    }
  });

  it("throws on missing ] in lvalue multi-assign output args", () => {
    try {
      parseMFile(
        "classdef C\n  methods (Abstract)\n    [a, b = foo(obj)\n  end\nend"
      );
    } catch (e) {
      expect(e).toBeInstanceOf(SyntaxError);
    }
  });

  it("throws on missing = after lvalue multi-assign output args", () => {
    try {
      parseMFile(
        "classdef C\n  methods (Abstract)\n    [a] foo(obj)\n  end\nend"
      );
    } catch (e) {
      expect(e).toBeInstanceOf(SyntaxError);
    }
  });
});

// ── Command form edge cases ──────────────────────────────────────────

describe("parseMFile - command edge cases", () => {
  it("parses command with end as argument", () => {
    const stmt = parseFirst("clear end\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.args[0].type).toBe("Ident");
    }
  });

  it("parses command with global as argument", () => {
    const stmt = parseFirst("clear global\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.args.length).toBeGreaterThanOrEqual(1);
    }
  });

  it("parses command with persistent as argument", () => {
    const stmt = parseFirst("clear persistent\n");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.args.length).toBeGreaterThanOrEqual(1);
    }
  });

  it("does not use command form when next is LParen", () => {
    const stmt = parseFirst("hold(gca)\n");
    if (stmt.type === "ExprStmt") {
      expect(stmt.expr.type).toBe("FuncCall");
    }
  });

  it("does not use command form when next is assignment", () => {
    const stmt = parseFirst("x = 5\n");
    expect(stmt.type).toBe("Assign");
  });
});

// ── General command syntax ───────────────────────────────────────────

describe("parseMFile - general command syntax", () => {
  it("parses disp hello as command syntax with char vector", () => {
    const stmt = parseFirst("disp hello\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.expr.type).toBe("FuncCall");
      if (stmt.expr.type === "FuncCall") {
        expect(stmt.expr.name).toBe("disp");
        expect(stmt.expr.args).toHaveLength(1);
        expect(stmt.expr.args[0].type).toBe("Char");
        if (stmt.expr.args[0].type === "Char") {
          expect(stmt.expr.args[0].value).toBe("'hello'");
        }
      }
    }
  });

  it("parses compound args with dots (load durer.mat)", () => {
    const stmt = parseFirst("load durer.mat\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("load");
      expect(stmt.expr.args).toHaveLength(1);
      if (stmt.expr.args[0].type === "Char") {
        expect(stmt.expr.args[0].value).toBe("'durer.mat'");
      }
    }
  });

  it("parses path args (cd some/path)", () => {
    const stmt = parseFirst("cd some/path\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("cd");
      expect(stmt.expr.args).toHaveLength(1);
      if (stmt.expr.args[0].type === "Char") {
        expect(stmt.expr.args[0].value).toBe("'some/path'");
      }
    }
  });

  it("parses flag and multiple args (whos -file durer.mat X)", () => {
    const stmt = parseFirst("whos -file durer.mat X\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("whos");
      expect(stmt.expr.args).toHaveLength(3);
      expect(stmt.expr.args[0].type).toBe("Char");
      expect(stmt.expr.args[1].type).toBe("Char");
      expect(stmt.expr.args[2].type).toBe("Char");
      if (
        stmt.expr.args[0].type === "Char" &&
        stmt.expr.args[1].type === "Char" &&
        stmt.expr.args[2].type === "Char"
      ) {
        expect(stmt.expr.args[0].value).toBe("'-file'");
        expect(stmt.expr.args[1].value).toBe("'durer.mat'");
        expect(stmt.expr.args[2].value).toBe("'X'");
      }
    }
  });

  it("parses quoted string with spaces", () => {
    const stmt = parseFirst("disp 'hello world'\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("disp");
      expect(stmt.expr.args).toHaveLength(1);
      if (stmt.expr.args[0].type === "Char") {
        expect(stmt.expr.args[0].value).toBe("'hello world'");
      }
    }
  });

  it("parses numeric arg as char vector", () => {
    const stmt = parseFirst("isnumeric 500\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.name).toBe("isnumeric");
      expect(stmt.expr.args).toHaveLength(1);
      if (stmt.expr.args[0].type === "Char") {
        expect(stmt.expr.args[0].value).toBe("'500'");
      }
    }
  });

  it("does not use command form for expressions with operators", () => {
    const expr = parseExpr("a + b");
    expect(expr.type).toBe("Binary");
  });

  it("does not use command form for logical operators", () => {
    const expr = parseExpr("a && b");
    expect(expr.type).toBe("Binary");
  });

  it("does not use command form for element-wise operators", () => {
    const expr = parseExpr("a .* b");
    expect(expr.type).toBe("Binary");
  });

  it("does not use command form when next is assignment", () => {
    const stmt = parseFirst("x = 5\n");
    expect(stmt.type).toBe("Assign");
  });

  it("does not use command form when next is LParen", () => {
    const stmt = parseFirst("foo(1)\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.expr.type).toBe("FuncCall");
      if (stmt.expr.type === "FuncCall") {
        // Should be normal function call, not command syntax
        expect(stmt.expr.args[0].type).toBe("Number");
      }
    }
  });

  it("preserves existing COMMAND_VERBS behavior", () => {
    // hold on → String arg (keyword normalization)
    const stmt = parseFirst("hold on\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.args[0].type).toBe("String");
    }
  });

  it("preserves clear with Ident args", () => {
    // clear x y z → Ident args (Any normalization)
    const stmt = parseFirst("clear x y z\n");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt" && stmt.expr.type === "FuncCall") {
      expect(stmt.expr.args[0].type).toBe("Ident");
      expect(stmt.expr.args[1].type).toBe("Ident");
      expect(stmt.expr.args[2].type).toBe("Ident");
    }
  });

  it("handles semicolon termination", () => {
    const stmt = parseFirst("disp hello;");
    expect(stmt.type).toBe("ExprStmt");
    if (stmt.type === "ExprStmt") {
      expect(stmt.suppressed).toBe(true);
      if (stmt.expr.type === "FuncCall") {
        expect(stmt.expr.args).toHaveLength(1);
      }
    }
  });
});

// ── Two-pass function end detection ──────────────────────────────────

describe("parseMFile - two-pass function end", () => {
  it("handles function without end with nested control flow", () => {
    const ast = parseMFile(
      "function y = foo(x)\n  if x > 0\n    y = x;\n  else\n    y = -x;\n  end\n"
    );
    expect(ast.body[0].type).toBe("Function");
  });

  it("handles three functions without end", () => {
    const ast = parseMFile(
      "function a = f1(x)\n  a = x;\n\nfunction b = f2(x)\n  b = x;\n\nfunction c = f3(x)\n  c = x;\n"
    );
    expect(ast.body).toHaveLength(3);
  });

  it("handles functions with end (all have end)", () => {
    const ast = parseMFile(
      "function a = f1(x)\n  a = x;\nend\n\nfunction b = f2(x)\n  b = x;\nend\n"
    );
    expect(ast.body).toHaveLength(2);
  });

  it("throws on mixed end/no-end functions", () => {
    expect(() =>
      parseMFile(
        "function a = f1(x)\n  a = x;\nend\n\nfunction b = f2(x)\n  b = x;\n"
      )
    ).toThrow();
  });
});

// ── Additional expression edge cases ─────────────────────────────────

describe("parseMFile - expression edge cases", () => {
  it("parses matrix with whitespace and unary minus", () => {
    // Tests the inMatrixExpr whitespace heuristic
    const expr = parseExpr("[1 -2 3]");
    if (expr.type === "Tensor") {
      expect(expr.rows[0]).toHaveLength(3);
    }
  });

  it("parses matrix with imaginary literal in whitespace context", () => {
    // Tests the imaginary literal exception in matrix whitespace heuristic
    const expr = parseExpr("[1 -2i]");
    if (expr.type === "Tensor") {
      // "-2i" should be parsed as a single element (unary minus on 2*i)
      expect(expr.rows[0].length).toBeGreaterThanOrEqual(1);
    }
  });

  it("parses cell with trailing comma before semicolon", () => {
    const expr = parseExpr("{1, 2,; 3, 4}");
    if (expr.type === "Cell") {
      expect(expr.rows).toHaveLength(2);
    }
  });

  it("does not parse LParen as postfix in matrix context with space", () => {
    // In matrix context, (1 -2) (3 +4) should be two elements
    const expr = parseExpr("[(1) (2)]");
    if (expr.type === "Tensor") {
      expect(expr.rows[0]).toHaveLength(2);
    }
  });

  it("parses meta-class with package path stopping at uppercase", () => {
    const expr = parseExpr("?MyClass");
    expect(expr.type).toBe("MetaClass");
    if (expr.type === "MetaClass") {
      expect(expr.name).toBe("MyClass");
    }
  });
});

// ── ParserBase edge cases ────────────────────────────────────────────

describe("parseMFile - parser base edge cases", () => {
  it("handles member access with keyword (if, for, while, etc.)", () => {
    for (const kw of ["if", "for", "while", "switch", "try", "end"]) {
      const expr = parseExpr(`obj.${kw}`);
      expect(expr.type).toBe("Member");
      if (expr.type === "Member") {
        expect(expr.name).toBe(kw);
      }
    }
  });

  it("handles global with comma-separated names", () => {
    const stmt = parseFirst("global x, y, z");
    if (stmt.type === "Global") {
      expect(stmt.names[0]).toBe("x");
    }
  });

  it("parses persistent with comma-separated names", () => {
    const stmt = parseFirst("persistent a, b");
    if (stmt.type === "Persistent") {
      expect(stmt.names[0]).toBe("a");
    }
  });
});

// ── ClassParser additional edge cases ────────────────────────────────

describe("parseMFile - classdef edge cases", () => {
  it("parses classdef with properties that have no default", () => {
    const stmt = parseFirst(
      "classdef C\n  properties\n    x\n    y\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const props = stmt.members[0];
      if (props.type === "Properties") {
        expect(props.defaultValues[0]).toBeNull();
        expect(props.defaultValues[1]).toBeNull();
      }
    }
  });

  it("handles classdef with events containing multiple names", () => {
    const stmt = parseFirst(
      "classdef C\n  events\n    EventA\n    EventB\n    EventC\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      const events = stmt.members.find(m => m.type === "Events");
      if (events?.type === "Events") {
        expect(events.names).toEqual(["EventA", "EventB", "EventC"]);
      }
    }
  });

  it("handles classdef with multiple member blocks", () => {
    const stmt = parseFirst(
      "classdef C\n  properties\n    x\n  end\n  methods\n    function foo(obj)\n    end\n  end\n  events\n    MyEvent\n  end\nend"
    );
    if (stmt.type === "ClassDef") {
      expect(stmt.members).toHaveLength(3);
    }
  });

  it("parses import with Dot then Star separately", () => {
    // import pkg.* where .* might not be tokenized as DotStar
    const stmt = parseFirst("import pkg.*");
    if (stmt.type === "Import") {
      expect(stmt.wildcard).toBe(true);
    }
  });
});

// ── Name=Value syntax in function calls ──────────────────────────────

describe("parseMFile - Name=Value syntax", () => {
  it("desugars Name=Value to string and value args", () => {
    const expr = parseExpr("plot(x, y, LineWidth=2)");
    expect(expr.type).toBe("FuncCall");
    if (expr.type === "FuncCall") {
      expect(expr.name).toBe("plot");
      expect(expr.args).toHaveLength(4);
      expect(expr.args[0].type).toBe("Ident");
      expect(expr.args[1].type).toBe("Ident");
      // LineWidth desugared to 'LineWidth' string literal
      expect(expr.args[2].type).toBe("Char");
      if (expr.args[2].type === "Char") {
        expect(expr.args[2].value).toBe("'LineWidth'");
      }
      expect(expr.args[3].type).toBe("Number");
    }
  });

  it("handles multiple Name=Value pairs", () => {
    const expr = parseExpr("foo(Color='red', Size=10)");
    expect(expr.type).toBe("FuncCall");
    if (expr.type === "FuncCall") {
      expect(expr.args).toHaveLength(4);
      expect(expr.args[0].type).toBe("Char");
      if (expr.args[0].type === "Char") {
        expect(expr.args[0].value).toBe("'Color'");
      }
      expect(expr.args[1].type).toBe("Char");
      expect(expr.args[2].type).toBe("Char");
      if (expr.args[2].type === "Char") {
        expect(expr.args[2].value).toBe("'Size'");
      }
      expect(expr.args[3].type).toBe("Number");
    }
  });

  it("mixes positional and Name=Value args", () => {
    const expr = parseExpr("bar(1, 2, Opt=3)");
    expect(expr.type).toBe("FuncCall");
    if (expr.type === "FuncCall") {
      expect(expr.args).toHaveLength(4);
      expect(expr.args[0].type).toBe("Number");
      expect(expr.args[1].type).toBe("Number");
      expect(expr.args[2].type).toBe("Char");
      if (expr.args[2].type === "Char") {
        expect(expr.args[2].value).toBe("'Opt'");
      }
      expect(expr.args[3].type).toBe("Number");
    }
  });

  it("does not confuse == comparison with Name=Value", () => {
    const expr = parseExpr("foo(x == 1)");
    expect(expr.type).toBe("FuncCall");
    if (expr.type === "FuncCall") {
      // x == 1 is a single comparison expression, not Name=Value
      expect(expr.args).toHaveLength(1);
      expect(expr.args[0].type).toBe("Binary");
    }
  });
});
