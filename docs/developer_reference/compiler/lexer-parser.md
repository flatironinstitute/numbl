# Lexer and Parser

## Lexer

Turns source text into tokens. The rules are MATLAB's, which are not regular:

- **Whitespace and commas are sometimes significant** — the lexer must know whether it is inside a matrix literal `[ ... ]` or function call to decide if a space separates elements or just pads tokens.
- **Apostrophe disambiguation** — `'` can mean string literal or transpose depending on what precedes it.
- **Commands vs function calls** — `foo bar baz` is a _command-form_ call with string arguments when it appears as a statement, but `foo(bar, baz)` is an expression call.
- **Line continuation** (`...`) and end-of-line semantics (newline terminates a statement; `;` suppresses output).

The lexer is table-driven from a token configuration. Add or change tokens by updating that configuration rather than by hand-coding rules in multiple places.

## Parser

A recursive-descent parser, split into focused sub-parsers by category (expressions, statements, control flow, functions, classes, command form, argument blocks). Output is an `AbstractSyntaxTree` — an object wrapping the statement list: `{ body: Stmt[] }`.

### AST shape

- `Stmt` is the discriminated union of statements: assignment, expression-statement, if/for/while/switch/try, function/class declarations, etc. (Arguments blocks are not a `Stmt` variant — they are stored as an `argumentsBlocks: ArgumentsBlock[]` field on the `Function` node.)
- `Expr` is the discriminated union of expressions: literals (number, char, string), identifier, binary/unary, range, function or method call, indexing (parens) and cell indexing (braces), member access, lambda, matrix and cell literals, `:` (colon), `end`.
- Every node carries a `span` recording its source location. Spans drive error messages and diagnostics.

### What the parser does _not_ do

- No name resolution — identifiers are bare names at this stage. Resolution happens in the lowering context or the interpreter.
- No type inference — all of that lives in the JIT.
- No constant folding or other optimization.

The result is a faithful structural representation of the source, ready for the interpreter to walk or the JIT to lower.
