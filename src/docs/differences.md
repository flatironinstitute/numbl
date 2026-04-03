# Differences from MATLAB

numbl aims for MATLAB compatibility but has some behavioral differences.

## `evalin` / `assignin` Require `% external-access`

MATLAB maintains a runtime symbol table for every scope, so `evalin`/`assignin` can access any variable dynamically. numbl compiles variables to JavaScript declarations with no symbol table, so you must declare which variables may be accessed externally:

```matlab
function result = my_function()
    % external-access: x y
    x = 10;
    y = 20;
    some_helper();      % can do assignin('caller', 'x', 99)
    result = x + y;     % x may have changed
end
```

The directive goes in the function that **owns** the variable, not in the function that calls `evalin`/`assignin`. Multiple `% external-access` lines are allowed. The directive is a comment, so MATLAB ignores it -- code runs identically in both environments.

**Dynamic fallback:** If `evalin`/`assignin` references a variable not in `% external-access`, numbl stores it in a separate dynamic map. These dynamic variables are only accessible through `evalin`/`assignin`, not as bare variable names.

Most MATLAB code does not use `evalin`/`assignin` and needs no directives.

## Numeric Types

All numeric values are double-precision. Single-precision (`single`) and integer types (`int8`, `uint16`, etc.) are not supported. Integer type names are recognized only in `fread`/`fwrite` format strings.

## Performance

Browser mode is designed for development, teaching, and sharing. The CLI with the native addon is significantly faster but still generally slower than MATLAB. Performance is actively improving.

## Unsupported Features

- Single-precision and integer numeric types
- Parallel computing (`parfor`, `spmd`)
- GPU arrays
- MEX interface
- Simulink and toolboxes
- Java / .NET integration
- App Designer / GUI
- Metaclasses and advanced class introspection
