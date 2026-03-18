# Differences between Numbl and MATLAB

Numbl compiles MATLAB to JavaScript ahead of time, so some features that rely on runtime introspection need explicit hints.

## `evalin` / `assignin` require `% external-access`

MATLAB maintains a runtime symbol table for every scope, so `evalin`/`assignin` can access any variable dynamically. Numbl compiles variables to JS `var` declarations with no symbol table, so you must declare which variables may be accessed externally:

```matlab
function result = my_function()
    % external-access: x y
    x = 10;
    y = 20;
    some_helper();      % can do assignin('caller', 'x', 99)
    result = x + y;     % x may have changed
end
```

The directive goes in the function that **owns** the variable, not in the function that calls `evalin`/`assignin`. For scripts, it declares workspace-accessible variables. Multiple `% external-access` lines are allowed. The directive is a comment, so MATLAB ignores it — code runs identically in both.

The directive registers getter/setter accessors for the listed variables, pre-declares variables that may be created purely via external `assignin`, and resets their inferred types after every function call to prevent incorrect compiler optimizations.

**Dynamic fallback:** If `evalin`/`assignin` references a variable _not_ in `% external-access`, Numbl stores it in a separate dynamic map (per-workspace or per-call-frame). These dynamic variables are only accessible through `evalin`/`assignin`, not as bare variable names — unlike MATLAB where they would appear in the local workspace.

Most MATLAB code does not use `evalin`/`assignin` and needs no directives.
