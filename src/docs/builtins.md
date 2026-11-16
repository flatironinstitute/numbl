# Built-in Functions

numbl includes more than 600 built-in functions. Run `numbl list-builtins` on the command line for the full list.

## Trigonometric

sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, sind, cosd, tand, sec, csc, cot, and their inverses.

## Exponential & Logarithmic

exp, log, log2, log10, log1p, pow2, sqrt

## Complex Numbers

real, imag, conj, angle, complex, hypot

## Rounding & Absolute Value

abs, floor, ceil, round, fix, sign

## Special Functions

erf, erfc, erfinv, erfcinv, gamma, gammaln, beta, airy, bessel functions, legendre, ellipj

## Array Construction

zeros, ones, eye, rand, randi, randn, randperm, linspace, logspace, colon

## Array Manipulation

reshape, squeeze, permute, repmat, repelem, cat, horzcat, vertcat, flip, fliplr, flipud, rot90, circshift

## Array Queries

size, length, numel, ndims, isempty, isscalar, isvector, ismatrix

## Reductions

sum, prod, mean, median, std, var, min, max, all, any, cumsum, cumprod, cummax, cummin

## Linear Algebra

inv, pinv, det, trace, rank, cond, rcond, norm, vecnorm, eig, eigs, svd, lu, qr, qz, chol, expm, kron, linsolve, mldivide, mrdivide, gmres, null, dot, cross, tril, triu, blkdiag, toeplitz, decomposition, isIllConditioned, istriu, istril, isbanded, bandwidth

## FFT

fft, ifft, fftshift, ifftshift

## Polynomials

poly, polyfit, polyval, roots, conv, deconv, mkpp, ppval

## Set Operations

unique, union, intersect, setdiff, ismember, uniquetol

## Sorting

sort, sortrows, mode

## String Operations

sprintf, strcmp, strcmpi, strncmp, strncmpi, strfind, strrep, strsplit, strjoin, strtrim, strip, strcat, strtok, strlength, deblank, blanks, newline, upper, lower, contains, startsWith, endsWith, replace, erase, count, reverse, pad, split, join, extractBefore, extractAfter, extractBetween, insertBefore, insertAfter, regexp, regexpi, regexprep

## Type Checking

isnumeric, isfloat, isinteger, islogical, ischar, isstring, iscell, isstruct, isreal, isfinite, isinf, isnan, issparse

## Type Conversion

double, single, logical, char, string, cellstr, cast, typecast, num2str, str2double, str2num, int2str, mat2str, dec2bin, bin2dec, dec2hex, hex2dec, func2str, str2func. `single` gives a distinct single-precision class; `int8`, ..., `uint64` round and saturate but return doubles (see [Differences from MATLAB](/docs/differences)).

## Data Validation

mustBeNumeric, mustBeFinite, mustBeInteger, mustBePositive, mustBeNonempty, mustBeInRange, mustBeMember, mustBeVector and the other `mustBe*` functions, validateattributes, narginchk, nargoutchk, inputParser (addRequired, addOptional, addParameter, parse)

## Sparse Matrices

sparse, speye, spdiags, spconvert, spalloc, full, nnz, nonzeros, sprand, sprandn, sprandsym, symrcm, amd, colamd. `chol`, `lu`, `kron`, `inv` and `\` work directly on sparse matrices (see [Differences from MATLAB](/docs/differences)).

## Struct & Cell

fieldnames, rmfield, cell2mat, cell2struct, struct2cell, num2cell, mat2cell, deal

## Interpolation & Grids

interp1, meshgrid, ndgrid

## Geometry

delaunay, delaunayn, triangulation, convhull, convhulln, inpolygon, cart2pol, pol2cart, cart2sph, sph2cart

## Graphs

graph, addedge, numnodes, numedges, degree, conncomp, laplacian

## Numerical Calculus

diff, gradient, trapz, cumtrapz, quadgk

## ODE Solvers

ode45, ode23, odeset, odeget, deval

## File I/O

fopen, fclose, fread, fwrite, fgetl, fgets, fscanf, sscanf, fileread, readmatrix, feof, fseek, ftell, dir, mkdir, delete, rmdir, copyfile, movefile, exist, isfile, isfolder, pwd, cd, fileparts, fullfile, tempdir, tempname, zip, unzip

## Web I/O

webread, websave, weboptions, jsonencode, jsondecode

## Formatting & Display

disp, fprintf, sprintf, warning, error, assert

## Timing

tic, toc, clock, etime

## Dates & Times

datetime, datestr, now, and the durations days, hours, minutes, seconds

## Dynamic Evaluation

eval, evalin, assignin, feval, builtin

## Higher-Order Functions

arrayfun, cellfun, structfun, bsxfun

## Batch Operations

pagemtimes, pagetranspose

## Dictionary

dictionary, keys, values, entries, lookup, insert, remove, isKey, isConfigured, numEntries, configureDictionary, types

## GPU Arrays

gpuArray, gather, gpuDevice, gpuDeviceCount, canUseGPU, existsOnGPU, isgpuarray, underlyingType

`gpuArray(X)` moves an array to a GPU, where operators and functions run on the device and return gpuArrays; `gather` brings the result back. In the browser numbl uses WebGPU. On the command line it uses WebGPU when the optional `webgpu` npm package is installed (`npm install -g webgpu`; set `NUMBL_NO_WEBGPU` to turn it off), and otherwise CPU devices that run the same code without a speedup. WebGPU computes in single precision only, so make GPU arrays from single data: `gpuArray(single(X))`. Element-wise operations, reductions, matrix products, indexing, `sort` and `fft` run on the GPU; other functions copy their arguments back to the host. Set `NUMBL_GPU_LOG` to see where that happens.
