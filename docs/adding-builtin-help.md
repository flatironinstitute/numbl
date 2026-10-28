# Adding Help Text for Builtins

This guide explains how to add help text so that `help <name>` works for a builtin.

## Quick check

Run `npx tsx src/cli.ts list-builtins --no-help` to see which builtins still need help text.

## Structure

Each help entry has two fields:

```typescript
interface BuiltinHelp {
  signatures: string[]; // calling forms, e.g. ["Y = sin(X)"]
  description: string; // one or two sentences
}
```

## Where to add help

There are two ways, depending on whether you're adding a new builtin or documenting an existing one.

### New builtins (preferred)

Pass `help` inline when calling `builtin`:

```typescript
builtin({
  name: "myFunc",
  help: {
    signatures: ["Y = myFunc(X)", "Y = myFunc(X, N)"],
    description: "Does something useful, element-wise.",
  },
  nargin: [1, 2],
  apply: args => ...,
});
```

The help is stored on the registered `IBuiltin` (`src/numbl-core/runtime/builtinRegistry.ts`).

### Existing builtins (bulk)

Add an entry to the `H` record in `src/numbl-core/interpreter/builtins/help-text.ts`:

```typescript
myFunc: {
  signatures: ["Y = myFunc(X)", "Y = myFunc(X, N)"],
  description: "Does something useful, element-wise.",
},
```

This file calls `registerBuiltinHelp()` to fill a separate help registry, which `getIBuiltinHelp()` consults when the builtin has no inline `help`.

## Writing good help text

### Signatures

- List every supported calling form. Check the implementation (`nargin` and `apply`); don't guess.
- Use the conventional output variable names: `Y` for general output, `TF` for logical, `[M, I]` for value+index, etc.
- Include multi-output forms: `[U, S, V] = svd(A)`.
- Include optional arguments as separate signatures rather than brackets: write two lines `sort(A)` and `sort(A, DIM)` instead of `sort(A [, DIM])`.
- For operator equivalents, include them: `"B = A.'"` alongside `"B = transpose(A)"`.

### Description

- Keep it to one or two sentences.
- Describe what the function does, not how it works internally.
- Mention non-obvious behavior: "Returns complex for negative input", "Ties round away from zero".
- Don't copy from external documentation. Base descriptions on what the numbl implementation actually does.

### Verifying accuracy

Before submitting, verify each entry against the source:

1. Read the `nargin` metadata and `apply` to confirm which argument patterns are accepted.
2. Read the `apply` function to confirm behavior (return types, special cases).
3. Test with `npx tsx src/cli.ts eval "help myFunc"` to check formatting.

### What NOT to document

- Don't document argument patterns that aren't implemented, even if the corresponding external tool supports them.
- Don't list every type the function accepts (number, tensor, sparse, etc.); the signatures should show the calling forms, not the type dispatch.
