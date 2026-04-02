# MATLAB Compatibility

Numbl implements a substantial subset of the MATLAB language. This page summarizes what is and isn't supported.

## Builtin Functions

More than 400 builtin functions are available. Run `numbl list-builtins` for the full list. Categories include:

| Category                  | Examples                                                                                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trigonometric             | sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, sind, cosd, tand, sec, csc, cot, and inverses                                                  |
| Exponential & logarithmic | exp, log, log2, log10, log1p, pow2, sqrt                                                                                                                 |
| Complex numbers           | real, imag, conj, angle, complex, hypot                                                                                                                  |
| Rounding                  | abs, floor, ceil, round, fix, sign                                                                                                                       |
| Special functions         | erf, erfc, erfinv, erfcinv, gamma, gammaln, beta, airy, bessel\*, legendre, ellipj                                                                       |
| Array construction        | zeros, ones, eye, rand, randi, randn, randperm, linspace, logspace, colon                                                                                |
| Array manipulation        | reshape, squeeze, permute, repmat, repelem, cat, horzcat, vertcat, flip, fliplr, flipud, rot90, circshift                                                |
| Array queries             | size, length, numel, ndims, isempty, isscalar, isvector, ismatrix                                                                                        |
| Reductions                | sum, prod, mean, median, std, var, min, max, all, any, cumsum, cumprod, cummax, cummin                                                                   |
| Linear algebra            | inv, pinv, det, trace, rank, cond, norm, eig, svd, lu, qr, qz, chol, linsolve, mldivide, mrdivide                                                        |
| FFT                       | fft, ifft, fftshift, ifftshift                                                                                                                           |
| Polynomials               | poly, polyfit, polyval, roots, conv, deconv                                                                                                              |
| Set operations            | unique, union, intersect, setdiff, ismember, uniquetol                                                                                                   |
| Sorting                   | sort, sortrows, mode                                                                                                                                     |
| String operations         | sprintf, strcmp, strcmpi, strfind, strrep, strsplit, strjoin, strtrim, upper, lower, contains, startsWith, endsWith, replace, regexp, regexpi, regexprep |
| Type checking             | isnumeric, isfloat, isinteger, islogical, ischar, isstring, iscell, isstruct, isreal, isfinite, isinf, isnan, issparse                                   |
| Type conversion           | double, logical, char, string, num2str, str2double, str2num, int2str                                                                                     |
| Data validation           | mustBeNumeric, mustBeFinite, mustBeInteger, mustBePositive, mustBeNonempty, mustBeInRange, mustBeMember, mustBeVector                                    |
| Sparse matrices           | sparse, speye, spdiags, spconvert, full, nnz, nonzeros                                                                                                   |
| Struct & cell             | fieldnames, rmfield, cell2mat, cell2struct, struct2cell, num2cell, mat2cell, deal                                                                        |
| Interpolation & grids     | interp1, meshgrid, ndgrid                                                                                                                                |
| Numerical calculus        | diff, gradient, trapz, cumtrapz                                                                                                                          |
| ODE solvers               | ode45, ode23, odeset, odeget, deval                                                                                                                      |
| File I/O                  | fopen, fclose, fread, fwrite, fgetl, fgets, fileread, feof, fseek, ftell, dir, mkdir, delete, rmdir, fileparts, fullfile, tempdir, tempname              |
| Web I/O                   | websave, webread                                                                                                                                         |
| Formatting & display      | disp, fprintf, sprintf, warning, error, assert                                                                                                           |
| Timing                    | tic, toc, clock, etime                                                                                                                                   |
| Dynamic evaluation        | eval, evalin, assignin, feval, builtin                                                                                                                   |
| Higher-order functions    | arrayfun, cellfun, structfun, bsxfun                                                                                                                     |
| Batch operations          | pagemtimes, pagetranspose                                                                                                                                |
| Dictionary                | dictionary, keys, values, entries, lookup, insert, remove, isKey, isConfigured, numEntries, configureDictionary, types                                   |

## Plotting

| Type              | Functions                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Line              | plot, plot3, fplot, fplot3                                                                                                                       |
| Scatter           | scatter, scatter3                                                                                                                                |
| Bar               | bar, barh, bar3, bar3h                                                                                                                           |
| Surface           | surf, mesh                                                                                                                                       |
| Contour           | contour                                                                                                                                          |
| Image             | imagesc                                                                                                                                          |
| Histogram         | histogram, histogram2                                                                                                                            |
| Statistical       | boxchart, swarmchart, swarmchart3                                                                                                                |
| Other             | stairs, errorbar, area, semilogx, semilogy, loglog, piechart, donutchart, heatmap                                                                |
| Figure management | figure, subplot, title, xlabel, ylabel, zlabel, sgtitle, legend, hold, grid, axis, view, colormap, colorbar, shading, close, clf, drawnow, pause |

## Language Features

| Feature                   | Status                                                                           |
| ------------------------- | -------------------------------------------------------------------------------- |
| Arithmetic operators      | `+` `-` `*` `/` `\` `^` `.*` `./` `.\` `.^`                                      |
| Comparison operators      | `==` `~=` `<` `<=` `>` `>=`                                                      |
| Logical operators         | `&&` `\|\|` `&` `\|` `~` `xor`                                                   |
| Transpose                 | `'` `.'`                                                                         |
| Matrix literals           | `[a b; c d]`                                                                     |
| Colon ranges              | `a:b`, `a:b:c`                                                                   |
| `end` indexing            | `A(end)`, `A(end-1, :)`                                                          |
| Logical indexing          | `A(A > 0)`                                                                       |
| Cell arrays               | `{...}` construction and `{...}` indexing                                        |
| Structs                   | dot access, dynamic field `s.(name)`                                             |
| if / elseif / else        | Supported                                                                        |
| for / while               | Supported                                                                        |
| switch / case / otherwise | Supported                                                                        |
| try / catch               | Supported                                                                        |
| break / continue / return | Supported                                                                        |
| Function definitions      | Regular, anonymous `@(x) ...`, handles `@func`, nested, subfunctions             |
| Multiple return values    | `[a, b] = func(...)`                                                             |
| varargin / varargout      | Supported                                                                        |
| nargin / nargout          | Supported                                                                        |
| Classes (classdef)        | Properties, methods, inheritance, static methods, abstract classes, enumerations |
| Global variables          | `global` keyword                                                                 |
| Persistent variables      | `persistent` keyword                                                             |
| Sparse matrices           | Full arithmetic support (CSC format)                                             |
| Complex numbers           | Full support throughout                                                          |
| Regular expressions       | regexp, regexpi, regexprep                                                       |
| String and char types     | Both `"string"` and `'char'` literals as distinct types                          |
| Comments                  | `%` line comments, `%{ %}` block comments                                        |

## Data Types

| Type                             | Status                                                                        |
| -------------------------------- | ----------------------------------------------------------------------------- |
| double                           | Supported (default numeric type)                                              |
| single                           | Not supported (all numerics are double precision)                             |
| logical                          | Supported                                                                     |
| char                             | Supported                                                                     |
| string                           | Supported                                                                     |
| int8 / int16 / int32 / int64     | Not supported (integer precision strings are recognized in fread/fwrite only) |
| uint8 / uint16 / uint32 / uint64 | Not supported (integer precision strings are recognized in fread/fwrite only) |
| complex                          | Supported                                                                     |
| cell                             | Supported                                                                     |
| struct                           | Supported                                                                     |
| sparse                           | Supported (real and complex, CSC format)                                      |
| function_handle                  | Supported                                                                     |
| class instances                  | Supported (value and handle classes)                                          |
| dictionary                       | Supported (MATLAB R2022b+ feature)                                            |

## Not Supported

Notable MATLAB features that are not yet implemented:

- Single-precision and integer numeric types
- Parallel computing (parfor, spmd)
- GPU arrays
- MEX interface
- Simulink
- Toolboxes
- Java / .NET integration
- App Designer / GUI
- Metaclasses and advanced class introspection

## Behavioral Differences from MATLAB

### `evalin` / `assignin` require `% external-access`

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
