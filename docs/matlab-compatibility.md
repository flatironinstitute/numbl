# MATLAB Compatibility

Numbl implements a substantial subset of the MATLAB language. This page summarizes what is and isn't supported.

## Builtin Functions

More than 400 builtin functions are available. Run `numbl list-builtins` for the full list. Categories include:

| Category                  | Examples                                                                                                                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trigonometric             | sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, sind, cosd, tand, sec, csc, cot, and inverses                                                                                        |
| Exponential & logarithmic | exp, log, log2, log10, log1p, pow2, sqrt                                                                                                                                                       |
| Complex numbers           | real, imag, conj, angle, complex, hypot                                                                                                                                                        |
| Rounding                  | abs, floor, ceil, round, fix, sign                                                                                                                                                             |
| Special functions         | erf, erfc, erfinv, erfcinv, gamma, gammaln, beta, airy, bessel\*, legendre, ellipj                                                                                                             |
| Array construction        | zeros, ones, eye, rand, randi, randn, randperm, linspace, logspace, colon                                                                                                                      |
| Array manipulation        | reshape, squeeze, permute, repmat, repelem, cat, horzcat, vertcat, flip, fliplr, flipud, rot90, circshift                                                                                      |
| Array queries             | size, length, numel, ndims, isempty, isscalar, isvector, ismatrix                                                                                                                              |
| Reductions                | sum, prod, mean, median, std, var, min, max, all, any, cumsum, cumprod, cummax, cummin                                                                                                         |
| Linear algebra            | inv, pinv, det, trace, rank, cond, norm, eig (see note), svd, lu, qr, qz, chol, expm, kron, linsolve, mldivide, mrdivide, decomposition, isIllConditioned, istriu, istril, isbanded, bandwidth |
| FFT                       | fft, ifft, fftshift, ifftshift                                                                                                                                                                 |
| Polynomials               | poly, polyfit, polyval, roots, conv, deconv                                                                                                                                                    |
| Set operations            | unique, union, intersect, setdiff, ismember, uniquetol                                                                                                                                         |
| Sorting                   | sort, sortrows, mode                                                                                                                                                                           |
| String operations         | sprintf, strcmp, strcmpi, strfind, strrep, strsplit, strjoin, strtrim, upper, lower, contains, startsWith, endsWith, replace, regexp, regexpi, regexprep                                       |
| Type checking             | isnumeric, isfloat, isinteger, islogical, ischar, isstring, iscell, isstruct, isreal, isfinite, isinf, isnan, issparse                                                                         |
| Type conversion           | double, logical, char, string, num2str, str2double, str2num, int2str                                                                                                                           |
| Data validation           | mustBeNumeric, mustBeFinite, mustBeInteger, mustBePositive, mustBeNonempty, mustBeInRange, mustBeMember, mustBeVector                                                                          |
| Sparse matrices           | sparse, speye, spdiags, spconvert, spalloc, full, nnz, nonzeros, sprand, sprandn, sprandsym, symrcm, amd, colamd, sparse chol / lu / kron / inv / mldivide (see note)                          |
| Struct & cell             | fieldnames, rmfield, cell2mat, cell2struct, struct2cell, num2cell, mat2cell, deal                                                                                                              |
| Interpolation & grids     | interp1, meshgrid, ndgrid                                                                                                                                                                      |
| Numerical calculus        | diff, gradient, trapz, cumtrapz                                                                                                                                                                |
| ODE solvers               | ode45, ode23, odeset, odeget, deval                                                                                                                                                            |
| File I/O                  | fopen, fclose, fread, fwrite, fgetl, fgets, fileread, feof, fseek, ftell, dir, mkdir, delete, rmdir, fileparts, fullfile, tempdir, tempname                                                    |
| Web I/O                   | websave, webread                                                                                                                                                                               |
| Formatting & display      | disp, fprintf, sprintf, warning, error, assert                                                                                                                                                 |
| Timing                    | tic, toc, clock, etime                                                                                                                                                                         |
| Dynamic evaluation        | eval, evalin, assignin, feval, builtin                                                                                                                                                         |
| Higher-order functions    | arrayfun, cellfun, structfun, bsxfun                                                                                                                                                           |
| Batch operations          | pagemtimes, pagetranspose                                                                                                                                                                      |
| Dictionary                | dictionary, keys, values, entries, lookup, insert, remove, isKey, isConfigured, numEntries, configureDictionary, types                                                                         |

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

| Feature                   | Status                                                                                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arithmetic operators      | `+` `-` `*` `/` `\` `^` `.*` `./` `.\` `.^`                                                                                                                                                                               |
| Comparison operators      | `==` `~=` `<` `<=` `>` `>=`                                                                                                                                                                                               |
| Logical operators         | `&&` `\|\|` `&` `\|` `~` `xor`                                                                                                                                                                                            |
| Transpose                 | `'` `.'`                                                                                                                                                                                                                  |
| Matrix literals           | `[a b; c d]`                                                                                                                                                                                                              |
| Colon ranges              | `a:b`, `a:b:c`                                                                                                                                                                                                            |
| `end` indexing            | `A(end)`, `A(end-1, :)`                                                                                                                                                                                                   |
| Logical indexing          | `A(A > 0)`                                                                                                                                                                                                                |
| Cell arrays               | `{...}` construction and `{...}` indexing                                                                                                                                                                                 |
| Structs                   | dot access, dynamic field `s.(name)`                                                                                                                                                                                      |
| if / elseif / else        | Supported                                                                                                                                                                                                                 |
| for / while               | Supported                                                                                                                                                                                                                 |
| switch / case / otherwise | Supported                                                                                                                                                                                                                 |
| try / catch               | Supported                                                                                                                                                                                                                 |
| break / continue / return | Supported                                                                                                                                                                                                                 |
| Function definitions      | Regular, anonymous `@(x) ...`, handles `@func`, nested, subfunctions                                                                                                                                                      |
| Multiple return values    | `[a, b] = func(...)`                                                                                                                                                                                                      |
| varargin / varargout      | Supported                                                                                                                                                                                                                 |
| nargin / nargout          | Supported                                                                                                                                                                                                                 |
| Classes (classdef)        | Properties, methods, inheritance, static methods, handle vs. value classes, abstract method signatures (parsed; no subclass-coverage enforcement). Enumeration blocks are parsed but not yet wired into runtime behavior. |
| Global variables          | `global` keyword                                                                                                                                                                                                          |
| Persistent variables      | `persistent` keyword                                                                                                                                                                                                      |
| Sparse matrices           | Full arithmetic support (CSC format); sparse direct solvers (Cholesky, LU with partial pivoting, triangular solves) for `\`, `/`, `chol`, `lu`, `inv`, `decomposition`                                                    |
| Complex numbers           | Full support throughout                                                                                                                                                                                                   |
| Regular expressions       | regexp, regexpi, regexprep                                                                                                                                                                                                |
| String and char types     | Both `"string"` and `'char'` literals as distinct types                                                                                                                                                                   |
| Comments                  | `%` line comments, `%{ %}` block comments                                                                                                                                                                                 |

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

### `eig`

`eig(A)` follows MATLAB's dispatch: exactly symmetric (or Hermitian) input goes to the symmetric eigensolver, giving real eigenvalues in ascending order and an orthonormal eigenvector basis; everything else goes to the general solver. Symmetry is tested with exact equality, as MATLAB's `isequal(A, A')` does, so a matrix that is symmetric only to within rounding takes the general path in both.

`eig(A, B)` solves the generalized problem `A*v = lambda*B*v`. A symmetric/Hermitian-definite pencil is reduced through the Cholesky factor of `B` and normalized so that `V'*B*V = I`, matching MATLAB. Any other pencil is solved by QZ, which requires the native LAPACK addon; without it (in the browser, or where the addon is not built) `eig(A, B)` raises an error rather than falling back to a different algorithm. The `'chol'`, `'qz'`, `'vector'`, `'matrix'`, `'balance'` and `'nobalance'` options are supported.

### Sparse direct solvers

Sparse matrices are factored without converting to full storage. `chol(S)` is an up-looking sparse Cholesky that, like MATLAB, reads only the upper triangle and does not check symmetry; `[R, p] = chol(S)` reports the failing column and returns the first `p-1` rows of the factor. The three-output form `[R, p, P] = chol(S)` applies a fill-reducing ordering. `lu(S)` is a left-looking LU with threshold partial pivoting (`[L, U, P]`, `[L, U, P, Q]`, the `'vector'` form, and a numeric pivot threshold are supported; the threshold defaults to 1 for three outputs and 0.1 for four). `S \ b` detects triangular matrices structurally and solves them by substitution, tries a permuted Cholesky when `S` is symmetric with a positive diagonal, and otherwise uses LU with a column ordering; a nearly full sparse matrix (more than half nonzeros) is handed to the dense LAPACK path. `b / S` is solved through the transpose, and `inv(S)` returns a sparse result. `kron` with a sparse operand returns sparse. Complex sparse matrices and non-square (least-squares) sparse systems still go through the dense path. The orderings differ from MATLAB's: `amd` is an exact minimum degree ordering (MATLAB uses approximate minimum degree) and `colamd` is minimum degree on the pattern of `A'*A`, so the permutations are valid and fill-reducing but not identical to MATLAB's. `decomposition(A)` is a builtin object supporting the `'auto'`, `'chol'`, `'lu'` and `'triangular'` types (other MATLAB types fall back to LU); it requires a real square matrix, and for dense input the `'auto'` type is `'lu'` unless the matrix is triangular.

## Disclaimer

numbl is not affiliated with, endorsed by, or supported by MathWorks, Inc. MATLAB is a registered trademark of MathWorks, Inc.
