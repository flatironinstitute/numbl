# Built-in Functions

numbl includes more than 400 built-in functions. Run `numbl list-builtins` on the command line for the full list.

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

inv, pinv, det, trace, rank, cond, norm, eig, svd, lu, qr, qz, chol, linsolve, mldivide, mrdivide

## FFT

fft, ifft, fftshift, ifftshift

## Polynomials

poly, polyfit, polyval, roots, conv, deconv

## Set Operations

unique, union, intersect, setdiff, ismember, uniquetol

## Sorting

sort, sortrows, mode

## String Operations

sprintf, strcmp, strcmpi, strfind, strrep, strsplit, strjoin, strtrim, upper, lower, contains, startsWith, endsWith, replace, regexp, regexpi, regexprep

## Type Checking

isnumeric, isfloat, isinteger, islogical, ischar, isstring, iscell, isstruct, isreal, isfinite, isinf, isnan, issparse

## Type Conversion

double, logical, char, string, num2str, str2double, str2num, int2str

## Data Validation

mustBeNumeric, mustBeFinite, mustBeInteger, mustBePositive, mustBeNonempty, mustBeInRange, mustBeMember, mustBeVector

## Sparse Matrices

sparse, speye, spdiags, spconvert, full, nnz, nonzeros

## Struct & Cell

fieldnames, rmfield, cell2mat, cell2struct, struct2cell, num2cell, mat2cell, deal

## Interpolation & Grids

interp1, meshgrid, ndgrid

## Numerical Calculus

diff, gradient, trapz, cumtrapz

## ODE Solvers

ode45, ode23, odeset, odeget, deval

## File I/O

fopen, fclose, fread, fwrite, fgetl, fgets, fileread, feof, fseek, ftell, dir, mkdir, delete, rmdir, fileparts, fullfile, tempdir, tempname

## Web I/O

websave, webread

## Formatting & Display

disp, fprintf, sprintf, warning, error, assert

## Timing

tic, toc, clock, etime

## Dynamic Evaluation

eval, evalin, assignin, feval, builtin

## Higher-Order Functions

arrayfun, cellfun, structfun, bsxfun

## Batch Operations

pagemtimes, pagetranspose

## Dictionary

dictionary, keys, values, entries, lookup, insert, remove, isKey, isConfigured, numEntries, configureDictionary, types
