/**
 * Help text for builtins. Registered via registerBuiltinHelp so existing
 * builtin definitions don't need to be modified.
 */

import { registerBuiltinHelp, type BuiltinHelp } from "./types.js";

const H: Record<string, BuiltinHelp> = {
  // ── Math ──────────────────────────────────────────────────────────────
  sin: {
    signatures: ["Y = sin(X)"],
    description: "Sine of argument in radians, element-wise.",
  },
  cos: {
    signatures: ["Y = cos(X)"],
    description: "Cosine of argument in radians, element-wise.",
  },
  tan: {
    signatures: ["Y = tan(X)"],
    description: "Tangent of argument in radians, element-wise.",
  },
  asin: {
    signatures: ["Y = asin(X)"],
    description:
      "Inverse sine in radians, element-wise. Returns complex for |X| > 1.",
  },
  acos: {
    signatures: ["Y = acos(X)"],
    description:
      "Inverse cosine in radians, element-wise. Returns complex for |X| > 1.",
  },
  atan: {
    signatures: ["Y = atan(X)"],
    description: "Inverse tangent in radians, element-wise.",
  },
  sinh: {
    signatures: ["Y = sinh(X)"],
    description: "Hyperbolic sine, element-wise.",
  },
  cosh: {
    signatures: ["Y = cosh(X)"],
    description: "Hyperbolic cosine, element-wise.",
  },
  tanh: {
    signatures: ["Y = tanh(X)"],
    description: "Hyperbolic tangent, element-wise.",
  },
  exp: {
    signatures: ["Y = exp(X)"],
    description: "Exponential (e^X), element-wise.",
  },
  log: {
    signatures: ["Y = log(X)"],
    description:
      "Natural logarithm, element-wise. Returns complex for negative input.",
  },
  log2: {
    signatures: ["Y = log2(X)", "[F, E] = log2(X)"],
    description:
      "Base-2 logarithm. With two outputs, returns fraction F and exponent E such that X = F * 2^E.",
  },
  log10: {
    signatures: ["Y = log10(X)"],
    description:
      "Common (base-10) logarithm, element-wise. Returns complex for negative input.",
  },
  log1p: {
    signatures: ["Y = log1p(X)"],
    description: "log(1+X), accurate for small X.",
  },
  expm1: {
    signatures: ["Y = expm1(X)"],
    description: "exp(X)-1, accurate for small X.",
  },
  abs: {
    signatures: ["Y = abs(X)"],
    description:
      "Absolute value. For complex input, returns the magnitude. Always returns real.",
  },
  sqrt: {
    signatures: ["Y = sqrt(X)"],
    description:
      "Square root, element-wise. Returns complex for negative input.",
  },
  sign: {
    signatures: ["Y = sign(X)"],
    description:
      "Signum function. Returns -1, 0, or 1. For complex input, returns X/abs(X).",
  },
  round: {
    signatures: ["Y = round(X)", "Y = round(X, N)"],
    description:
      "Round to nearest integer (ties away from zero). With N, rounds to N decimal places.",
  },
  floor: {
    signatures: ["Y = floor(X)"],
    description: "Round toward negative infinity, element-wise.",
  },
  ceil: {
    signatures: ["Y = ceil(X)"],
    description: "Round toward positive infinity, element-wise.",
  },
  fix: {
    signatures: ["Y = fix(X)"],
    description: "Round toward zero, element-wise.",
  },
  hypot: {
    signatures: ["H = hypot(X, Y)"],
    description:
      "Square root of sum of squares: sqrt(X^2 + Y^2), element-wise.",
  },
  factorial: {
    signatures: ["Y = factorial(N)"],
    description: "Factorial of non-negative integer N, element-wise.",
  },
  nthroot: {
    signatures: ["Y = nthroot(X, N)"],
    description:
      "Real Nth root of X, element-wise. Unlike X^(1/N), returns real results for odd N and negative X.",
  },

  // ── Arithmetic ────────────────────────────────────────────────────────
  atan2: {
    signatures: ["T = atan2(Y, X)"],
    description: "Four-quadrant inverse tangent, in radians, element-wise.",
  },
  mod: {
    signatures: ["M = mod(A, B)"],
    description: "Modulus after division. Result has the sign of B.",
  },
  rem: {
    signatures: ["R = rem(A, B)"],
    description: "Remainder after division. Result has the sign of A.",
  },
  power: {
    signatures: ["Z = power(X, Y)", "Z = X .^ Y"],
    description: "Element-wise power.",
  },
  max: {
    signatures: [
      "M = max(A)",
      "[M, I] = max(A)",
      "M = max(A, B)",
      "M = max(A, [], DIM)",
      "[M, I] = max(A, [], DIM)",
      "M = max(A, [], 'all')",
    ],
    description:
      "Maximum value. With one array, reduces along the first non-singleton dimension. With two arrays, returns element-wise maximum.",
  },
  min: {
    signatures: [
      "M = min(A)",
      "[M, I] = min(A)",
      "M = min(A, B)",
      "M = min(A, [], DIM)",
      "[M, I] = min(A, [], DIM)",
      "M = min(A, [], 'all')",
    ],
    description:
      "Minimum value. With one array, reduces along the first non-singleton dimension. With two arrays, returns element-wise minimum.",
  },

  // ── Complex ───────────────────────────────────────────────────────────
  real: {
    signatures: ["Y = real(Z)"],
    description: "Real part of complex number or array.",
  },
  imag: {
    signatures: ["Y = imag(Z)"],
    description: "Imaginary part of complex number or array.",
  },
  conj: {
    signatures: ["Y = conj(Z)"],
    description: "Complex conjugate.",
  },
  angle: {
    signatures: ["T = angle(Z)"],
    description:
      "Phase angle in radians. Returns 0 for positive reals, pi for negative.",
  },
  complex: {
    signatures: ["Z = complex(A, B)", "Z = complex(A)"],
    description: "Construct complex number from real and imaginary parts.",
  },

  // ── Predicates ────────────────────────────────────────────────────────
  isnan: {
    signatures: ["TF = isnan(X)"],
    description: "True for elements that are NaN.",
  },
  isinf: {
    signatures: ["TF = isinf(X)"],
    description: "True for elements that are Inf or -Inf.",
  },
  isfinite: {
    signatures: ["TF = isfinite(X)"],
    description: "True for elements that are finite (not NaN or Inf).",
  },
  isreal: {
    signatures: ["TF = isreal(X)"],
    description:
      "True if X has no imaginary part (scalar result for entire array).",
  },

  // ── Introspection ─────────────────────────────────────────────────────
  size: {
    signatures: ["SZ = size(A)", "[M, N] = size(A)", "D = size(A, DIM)"],
    description: "Returns the dimensions of A.",
  },
  length: {
    signatures: ["L = length(A)"],
    description: "Length of largest dimension. Equivalent to max(size(A)).",
  },
  numel: {
    signatures: ["N = numel(A)"],
    description: "Total number of elements in A.",
  },
  ndims: {
    signatures: ["N = ndims(A)"],
    description: "Number of dimensions of A (always at least 2).",
  },
  isempty: {
    signatures: ["TF = isempty(A)"],
    description: "True if A is empty (has a zero dimension).",
  },
  isscalar: {
    signatures: ["TF = isscalar(A)"],
    description: "True if A is a 1x1 scalar.",
  },
  isvector: {
    signatures: ["TF = isvector(A)"],
    description: "True if A is a row or column vector.",
  },
  ismatrix: {
    signatures: ["TF = ismatrix(A)"],
    description: "True if A is a 2-D array.",
  },
  isnumeric: {
    signatures: ["TF = isnumeric(A)"],
    description: "True if A is a numeric type.",
  },
  islogical: {
    signatures: ["TF = islogical(A)"],
    description: "True if A is a logical array.",
  },
  ischar: {
    signatures: ["TF = ischar(A)"],
    description: "True if A is a character array.",
  },
  isstring: {
    signatures: ["TF = isstring(A)"],
    description: "True if A is a string.",
  },
  iscell: {
    signatures: ["TF = iscell(A)"],
    description: "True if A is a cell array.",
  },
  isstruct: {
    signatures: ["TF = isstruct(A)"],
    description: "True if A is a struct.",
  },
  issparse: {
    signatures: ["TF = issparse(A)"],
    description: "True if A is a sparse matrix.",
  },
  class: {
    signatures: ["C = class(A)"],
    description: "Returns the class name of A as a character vector.",
  },
  fieldnames: {
    signatures: ["F = fieldnames(S)"],
    description: "Returns cell array of field names of struct or object S.",
  },

  // ── Array construction ────────────────────────────────────────────────
  zeros: {
    signatures: [
      "X = zeros(N)",
      "X = zeros(M, N)",
      "X = zeros(M, N, P, ...)",
      "X = zeros([M, N])",
    ],
    description:
      "Create array of all zeros. Single arg N creates an N-by-N matrix.",
  },
  ones: {
    signatures: [
      "X = ones(N)",
      "X = ones(M, N)",
      "X = ones(M, N, P, ...)",
      "X = ones([M, N])",
    ],
    description:
      "Create array of all ones. Single arg N creates an N-by-N matrix.",
  },
  eye: {
    signatures: ["I = eye(N)", "I = eye(M, N)", "I = eye([M, N])"],
    description: "Identity matrix.",
  },
  linspace: {
    signatures: ["Y = linspace(A, B)", "Y = linspace(A, B, N)"],
    description:
      "Generate N linearly spaced points between A and B. Default N is 100.",
  },
  logspace: {
    signatures: ["Y = logspace(A, B)", "Y = logspace(A, B, N)"],
    description:
      "Generate N logarithmically spaced points between 10^A and 10^B. Default N is 50.",
  },
  rand: {
    signatures: [
      "X = rand",
      "X = rand(N)",
      "X = rand(M, N)",
      "X = rand(M, N, P, ...)",
    ],
    description:
      "Uniformly distributed random numbers in [0, 1). Single arg N creates an N-by-N matrix.",
  },
  randn: {
    signatures: [
      "X = randn",
      "X = randn(N)",
      "X = randn(M, N)",
      "X = randn(M, N, P, ...)",
    ],
    description:
      "Normally distributed random numbers (mean 0, std 1). Single arg N creates an N-by-N matrix.",
  },
  randi: {
    signatures: [
      "X = randi(IMAX)",
      "X = randi(IMAX, N)",
      "X = randi(IMAX, M, N)",
      "X = randi([IMIN IMAX], ...)",
    ],
    description: "Uniformly distributed random integers.",
  },
  randperm: {
    signatures: ["P = randperm(N)", "P = randperm(N, K)"],
    description:
      "Random permutation of integers 1:N. With K, returns K unique integers from 1:N.",
  },
  rng: {
    signatures: [
      "rng(SEED)",
      "rng('shuffle')",
      "rng('default')",
      "S = rng",
      "rng(S)",
    ],
    description:
      "Control random number generation. Set seed for reproducibility. With no args, returns current state as struct.",
  },

  // ── Array manipulation ────────────────────────────────────────────────
  reshape: {
    signatures: ["B = reshape(A, SZ)", "B = reshape(A, SZ1, SZ2, ...)"],
    description:
      "Reshape array to specified dimensions. Use [] for one dimension to auto-compute it.",
  },
  transpose: {
    signatures: ["B = transpose(A)", "B = A.'"],
    description: "Non-conjugate transpose.",
  },
  ctranspose: {
    signatures: ["B = ctranspose(A)", "B = A'"],
    description: "Complex conjugate transpose.",
  },
  diag: {
    signatures: ["D = diag(V)", "D = diag(V, K)", "V = diag(A)"],
    description:
      "If V is a vector, creates a diagonal matrix. If A is a matrix, extracts the diagonal. K offsets the diagonal.",
  },
  cat: {
    signatures: ["C = cat(DIM, A, B, ...)"],
    description: "Concatenate arrays along dimension DIM.",
  },
  horzcat: {
    signatures: ["C = horzcat(A, B, ...)", "C = [A, B, ...]"],
    description: "Horizontal concatenation.",
  },
  vertcat: {
    signatures: ["C = vertcat(A, B, ...)", "C = [A; B; ...]"],
    description: "Vertical concatenation.",
  },
  repmat: {
    signatures: [
      "B = repmat(A, N)",
      "B = repmat(A, M, N)",
      "B = repmat(A, [M, N])",
    ],
    description: "Replicate and tile an array.",
  },
  squeeze: {
    signatures: ["B = squeeze(A)"],
    description: "Remove singleton dimensions.",
  },
  flip: {
    signatures: ["B = flip(A)", "B = flip(A, DIM)"],
    description:
      "Flip array along specified dimension (default: first non-singleton).",
  },
  fliplr: {
    signatures: ["B = fliplr(A)"],
    description: "Flip array left to right.",
  },
  flipud: {
    signatures: ["B = flipud(A)"],
    description: "Flip array up to down.",
  },
  rot90: {
    signatures: ["B = rot90(A)", "B = rot90(A, K)"],
    description:
      "Rotate matrix 90 degrees counterclockwise K times (default K=1).",
  },
  circshift: {
    signatures: ["B = circshift(A, K)"],
    description:
      "Shift elements circularly. K is a scalar (shifts along first non-singleton dimension) or a vector (shifts along each dimension).",
  },
  permute: {
    signatures: ["B = permute(A, ORDER)"],
    description: "Rearrange dimensions of array.",
  },
  meshgrid: {
    signatures: [
      "[X, Y] = meshgrid(V)",
      "[X, Y] = meshgrid(XV, YV)",
      "[X, Y, Z] = meshgrid(XV, YV, ZV)",
    ],
    description: "Create 2-D or 3-D grid coordinates.",
  },
  ndgrid: {
    signatures: ["[X1, X2, ...] = ndgrid(V1, V2, ...)"],
    description: "Create N-D grid coordinates.",
  },
  sub2ind: {
    signatures: ["IND = sub2ind(SZ, I, J)", "IND = sub2ind(SZ, I1, I2, ...)"],
    description: "Convert subscripts to linear indices.",
  },
  ind2sub: {
    signatures: ["[I, J] = ind2sub(SZ, IND)"],
    description: "Convert linear indices to subscripts.",
  },
  repelem: {
    signatures: ["B = repelem(A, N)", "B = repelem(A, R, C)"],
    description:
      "Replicate elements of an array. With one count N, repeats each element N times. With two counts, repeats R times along rows and C times along columns.",
  },

  // ── Reductions ────────────────────────────────────────────────────────
  sum: {
    signatures: [
      "S = sum(A)",
      "S = sum(A, DIM)",
      "S = sum(A, 'all')",
      "S = sum(..., 'omitnan')",
    ],
    description: "Sum of elements along the specified dimension.",
  },
  prod: {
    signatures: [
      "P = prod(A)",
      "P = prod(A, DIM)",
      "P = prod(A, 'all')",
      "P = prod(..., 'omitnan')",
    ],
    description: "Product of elements along the specified dimension.",
  },
  mean: {
    signatures: [
      "M = mean(A)",
      "M = mean(A, DIM)",
      "M = mean(A, 'all')",
      "M = mean(..., 'omitnan')",
    ],
    description: "Average (mean) of elements along the specified dimension.",
  },
  median: {
    signatures: [
      "M = median(A)",
      "M = median(A, DIM)",
      "M = median(A, 'all')",
      "M = median(..., 'omitnan')",
    ],
    description: "Median value along the specified dimension.",
  },
  mode: {
    signatures: ["M = mode(A)", "M = mode(A, DIM)", "M = mode(A, 'all')"],
    description: "Most frequent value along the specified dimension.",
  },
  std: {
    signatures: [
      "S = std(A)",
      "S = std(A, W)",
      "S = std(A, W, DIM)",
      "S = std(A, W, 'all')",
    ],
    description:
      "Standard deviation. W=0 (default) normalizes by N-1, W=1 normalizes by N.",
  },
  var: {
    signatures: [
      "V = var(A)",
      "V = var(A, W)",
      "V = var(A, W, DIM)",
      "V = var(A, W, 'all')",
    ],
    description:
      "Variance. W=0 (default) normalizes by N-1, W=1 normalizes by N.",
  },
  any: {
    signatures: ["TF = any(A)", "TF = any(A, DIM)", "TF = any(A, 'all')"],
    description: "True if any element is nonzero.",
  },
  all: {
    signatures: ["TF = all(A)", "TF = all(A, DIM)", "TF = all(A, 'all')"],
    description: "True if all elements are nonzero.",
  },
  cumsum: {
    signatures: ["B = cumsum(A)", "B = cumsum(A, DIM)"],
    description: "Cumulative sum along the specified dimension.",
  },
  cumprod: {
    signatures: ["B = cumprod(A)", "B = cumprod(A, DIM)"],
    description: "Cumulative product along the specified dimension.",
  },
  cummax: {
    signatures: ["B = cummax(A)", "B = cummax(A, DIM)"],
    description: "Cumulative maximum along the specified dimension.",
  },
  cummin: {
    signatures: ["B = cummin(A)", "B = cummin(A, DIM)"],
    description: "Cumulative minimum along the specified dimension.",
  },
  diff: {
    signatures: ["D = diff(A)", "D = diff(A, N)", "D = diff(A, N, DIM)"],
    description:
      "Differences between adjacent elements. N specifies the order of difference.",
  },
  xor: {
    signatures: ["C = xor(A, B)"],
    description: "Logical exclusive OR.",
  },

  // ── Sorting & searching ───────────────────────────────────────────────
  sort: {
    signatures: [
      "B = sort(A)",
      "[B, I] = sort(A)",
      "B = sort(A, DIM)",
      "B = sort(A, 'descend')",
    ],
    description:
      "Sort elements in ascending order. Returns sorted array and optionally the sort indices.",
  },
  sortrows: {
    signatures: [
      "B = sortrows(A)",
      "[B, I] = sortrows(A)",
      "B = sortrows(A, COL)",
    ],
    description:
      "Sort rows of a matrix. Negative column indices sort descending.",
  },
  unique: {
    signatures: [
      "C = unique(A)",
      "[C, IA, IC] = unique(A)",
      "C = unique(A, 'rows')",
      "C = unique(A, 'stable')",
    ],
    description: "Unique values in sorted order.",
  },
  find: {
    signatures: [
      "I = find(X)",
      "[I, J] = find(X)",
      "[I, J, V] = find(X)",
      "I = find(X, N)",
      "I = find(X, N, 'last')",
    ],
    description: "Find indices of nonzero elements.",
  },
  intersect: {
    signatures: ["C = intersect(A, B)"],
    description: "Set intersection of two arrays.",
  },
  union: {
    signatures: ["C = union(A, B)"],
    description: "Set union of two arrays.",
  },
  setdiff: {
    signatures: ["C = setdiff(A, B)", "[C, IA] = setdiff(A, B)"],
    description: "Set difference: elements in A that are not in B.",
  },
  ismember: {
    signatures: ["TF = ismember(A, B)", "[TF, LOC] = ismember(A, B)"],
    description:
      "True for elements of A that are in B. LOC gives the index in B.",
  },
  nnz: {
    signatures: ["N = nnz(A)"],
    description: "Number of nonzero elements.",
  },

  // ── Linear algebra ────────────────────────────────────────────────────
  dot: {
    signatures: ["C = dot(A, B)"],
    description:
      "Dot product. For matrices, computes column-wise dot products.",
  },
  cross: {
    signatures: ["C = cross(A, B)", "C = cross(A, B, DIM)"],
    description:
      "Cross product of 3-element vectors. DIM specifies the dimension of length 3.",
  },
  norm: {
    signatures: ["N = norm(A)", "N = norm(A, P)"],
    description:
      "Vector or matrix norm. For vectors, P can be any number, Inf, or -Inf. For matrices, P can be 1, 2 (default), Inf, or 'fro'.",
  },
  det: {
    signatures: ["D = det(A)"],
    description: "Determinant of square matrix.",
  },
  inv: {
    signatures: ["B = inv(A)"],
    description: "Matrix inverse.",
  },
  trace: {
    signatures: ["T = trace(A)"],
    description: "Sum of diagonal elements.",
  },
  eig: {
    signatures: ["E = eig(A)", "[V, D] = eig(A)", "[V, D, W] = eig(A)"],
    description:
      "Eigenvalues and eigenvectors of square matrix. With three outputs, W contains left eigenvectors.",
  },
  svd: {
    signatures: [
      "S = svd(A)",
      "[U, S, V] = svd(A)",
      "[U, S, V] = svd(A, 'econ')",
      "[U, S, V] = svd(A, 0)",
    ],
    description: "Singular value decomposition.",
  },
  lu: {
    signatures: ["Y = lu(A)", "[L, U] = lu(A)", "[L, U, P] = lu(A)"],
    description:
      "LU factorization. With 2 outputs, L includes the permutation. With 3, P is a permutation matrix.",
  },
  qr: {
    signatures: [
      "[Q, R] = qr(A)",
      "[Q, R, E] = qr(A)",
      "[Q, R] = qr(A, 'econ')",
      "[Q, R, E] = qr(A, 'econ')",
    ],
    description:
      "QR factorization. With 3 outputs, E is a column permutation matrix (or vector in economy mode).",
  },
  chol: {
    signatures: ["R = chol(A)", "[R, P] = chol(A)"],
    description:
      "Cholesky factorization. A must be symmetric positive definite. With 2 outputs, P is 0 on success.",
  },
  rank: {
    signatures: ["K = rank(A)", "K = rank(A, TOL)"],
    description: "Matrix rank.",
  },
  pinv: {
    signatures: ["B = pinv(A)", "B = pinv(A, TOL)"],
    description: "Moore-Penrose pseudoinverse.",
  },
  cond: {
    signatures: ["C = cond(A)", "C = cond(A, P)"],
    description: "Condition number of matrix.",
  },
  kron: {
    signatures: ["K = kron(A, B)"],
    description: "Kronecker tensor product.",
  },
  linsolve: {
    signatures: ["X = linsolve(A, B)"],
    description: "Solve linear system A*X = B.",
  },

  // ── Strings ───────────────────────────────────────────────────────────
  sprintf: {
    signatures: ["S = sprintf(FMT, A, ...)"],
    description:
      "Format data into a string using format specifiers (%d, %f, %s, etc.).",
  },
  num2str: {
    signatures: [
      "S = num2str(A)",
      "S = num2str(A, FMT)",
      "S = num2str(A, PRECISION)",
    ],
    description:
      "Convert number to character array. FMT is a format string, PRECISION is number of significant digits.",
  },
  str2double: {
    signatures: ["X = str2double(S)", "X = str2double(C)"],
    description:
      "Convert string to double-precision number. Also accepts a cell array of strings. Returns NaN for non-numeric strings.",
  },
  str2num: {
    signatures: ["X = str2num(S)"],
    description:
      "Convert string to number. Returns empty matrix if conversion fails.",
  },
  strcmp: {
    signatures: ["TF = strcmp(S1, S2)"],
    description:
      "Compare strings (case sensitive). Also supports cell arrays for element-wise comparison.",
  },
  strcmpi: {
    signatures: ["TF = strcmpi(S1, S2)"],
    description:
      "Compare strings (case insensitive). Also supports cell arrays for element-wise comparison.",
  },
  strcat: {
    signatures: ["S = strcat(S1, S2, ...)"],
    description: "Concatenate strings horizontally.",
  },
  strsplit: {
    signatures: ["C = strsplit(S)", "C = strsplit(S, DELIM)"],
    description:
      "Split string at delimiters. DELIM can be a string or cell array of strings. Default is whitespace.",
  },
  strjoin: {
    signatures: ["S = strjoin(C)", "S = strjoin(C, DELIM)"],
    description:
      "Join cell array of strings with delimiter. Default delimiter is space.",
  },
  strlength: {
    signatures: ["L = strlength(S)"],
    description: "Number of characters in string.",
  },
  contains: {
    signatures: ["TF = contains(S, PAT)"],
    description:
      "True if string contains pattern. PAT can be a string or cell array of patterns.",
  },
  replace: {
    signatures: ["S2 = replace(S, OLD, NEW)"],
    description: "Replace all occurrences of OLD with NEW in string.",
  },
  upper: {
    signatures: ["S2 = upper(S)"],
    description: "Convert to uppercase.",
  },
  lower: {
    signatures: ["S2 = lower(S)"],
    description: "Convert to lowercase.",
  },
  strip: {
    signatures: ["S2 = strip(S)", "S2 = strip(S, SIDE)"],
    description:
      "Remove leading and trailing whitespace. SIDE can be 'left', 'right', or 'both' (default).",
  },
  regexp: {
    signatures: [
      "START = regexp(S, PAT)",
      "OUT = regexp(S, PAT, 'match')",
      "OUT = regexp(S, PAT, 'tokens')",
      "OUT = regexp(S, PAT, 'names')",
    ],
    description:
      "Match regular expression. Default returns start positions. Use 'match', 'tokens', 'names', 'start', 'end' to select output. 'once' returns only the first match.",
  },
  regexprep: {
    signatures: ["S2 = regexprep(S, PAT, REP)"],
    description: "Replace using regular expression.",
  },
  strfind: {
    signatures: ["K = strfind(S, PAT)"],
    description: "Find pattern in string. Returns starting indices.",
  },
  string: {
    signatures: ["S = string(X)"],
    description: "Convert to string type.",
  },
  char: {
    signatures: ["C = char(X)"],
    description:
      "Convert to character array. Numeric input is interpreted as character codes.",
  },

  // ── Type constructors ─────────────────────────────────────────────────
  double: {
    signatures: ["Y = double(X)"],
    description: "Convert to double precision.",
  },
  logical: {
    signatures: ["Y = logical(X)"],
    description: "Convert to logical (boolean) array.",
  },
  cell: {
    signatures: ["C = cell(N)", "C = cell(M, N)"],
    description: "Create cell array.",
  },
  struct: {
    signatures: [
      "S = struct()",
      "S = struct('field1', VAL1, 'field2', VAL2, ...)",
    ],
    description: "Create a struct with specified fields and values.",
  },
  deal: {
    signatures: ["[A, B, ...] = deal(X)", "[A, B, ...] = deal(X, Y, ...)"],
    description:
      "Distribute inputs to outputs. With one input, copies to all outputs.",
  },
  full: {
    signatures: ["A = full(S)"],
    description: "Convert sparse matrix to full matrix.",
  },
  sparse: {
    signatures: [
      "S = sparse(A)",
      "S = sparse(M, N)",
      "S = sparse(I, J, V)",
      "S = sparse(I, J, V, M, N)",
    ],
    description: "Create sparse matrix from dense array, size, or triplets.",
  },

  // ── Numerical ─────────────────────────────────────────────────────────
  interp1: {
    signatures: ["YI = interp1(X, Y, XI)", "YI = interp1(X, Y, XI, METHOD)"],
    description: "1-D interpolation. Methods: 'linear' (default), 'nearest'.",
  },
  polyval: {
    signatures: ["Y = polyval(P, X)"],
    description: "Evaluate polynomial P at points X.",
  },
  polyfit: {
    signatures: ["P = polyfit(X, Y, N)"],
    description: "Polynomial curve fitting of degree N.",
  },
  roots: {
    signatures: ["R = roots(P)"],
    description: "Find polynomial roots.",
  },
  poly: {
    signatures: ["P = poly(R)", "P = poly(A)"],
    description:
      "Polynomial with specified roots, or characteristic polynomial of matrix.",
  },
  conv: {
    signatures: ["C = conv(A, B)", "C = conv(A, B, SHAPE)"],
    description:
      "Convolution of two vectors. SHAPE: 'full' (default), 'same', 'valid'.",
  },
  trapz: {
    signatures: ["Q = trapz(Y)", "Q = trapz(X, Y)"],
    description: "Trapezoidal numerical integration.",
  },
  gradient: {
    signatures: ["FX = gradient(F)", "[FX, FY] = gradient(F)"],
    description: "Numerical gradient.",
  },
  eps: {
    signatures: ["E = eps", "E = eps(X)"],
    description: "Floating-point relative accuracy. eps(X) gives spacing at X.",
  },

  // ── FFT ───────────────────────────────────────────────────────────────
  fft: {
    signatures: ["Y = fft(X)", "Y = fft(X, N)", "Y = fft(X, N, DIM)"],
    description: "Discrete Fourier transform.",
  },
  ifft: {
    signatures: ["X = ifft(Y)", "X = ifft(Y, N)", "X = ifft(Y, N, DIM)"],
    description: "Inverse discrete Fourier transform.",
  },
  fftshift: {
    signatures: ["Y = fftshift(X)", "Y = fftshift(X, DIM)"],
    description: "Shift zero-frequency component to center of spectrum.",
  },
  ifftshift: {
    signatures: ["Y = ifftshift(X)", "Y = ifftshift(X, DIM)"],
    description: "Inverse of fftshift.",
  },

  // ── Utility ───────────────────────────────────────────────────────────
  assert: {
    signatures: ["assert(COND)", "assert(COND, MSG)"],
    description: "Throw error if condition is false.",
  },
  error: {
    signatures: ["error(MSG)", "error(ID, MSG, ...)"],
    description: "Throw an error with a message.",
  },
  isequal: {
    signatures: ["TF = isequal(A, B, ...)"],
    description: "True if all inputs are equal (NaN ~= NaN).",
  },

  // ── Misc ──────────────────────────────────────────────────────────────
  disp: {
    signatures: ["disp(X)"],
    description: "Display value of variable.",
  },
  fprintf: {
    signatures: ["fprintf(FMT, A, ...)", "fprintf(FID, FMT, A, ...)"],
    description: "Write formatted data to screen or file.",
  },
  tic: {
    signatures: ["tic"],
    description: "Start a stopwatch timer.",
  },
  toc: {
    signatures: ["toc", "T = toc"],
    description: "Read elapsed time from tic.",
  },

  // ── Bit operations ────────────────────────────────────────────────────
  bitand: {
    signatures: ["C = bitand(A, B)"],
    description: "Bitwise AND.",
  },
  bitor: {
    signatures: ["C = bitor(A, B)"],
    description: "Bitwise OR.",
  },
  bitxor: {
    signatures: ["C = bitxor(A, B)"],
    description: "Bitwise XOR.",
  },

  // ── Array extras ──────────────────────────────────────────────────────
  triu: {
    signatures: ["U = triu(A)", "U = triu(A, K)"],
    description: "Upper triangular part of matrix. K offsets the diagonal.",
  },
  tril: {
    signatures: ["L = tril(A)", "L = tril(A, K)"],
    description: "Lower triangular part of matrix. K offsets the diagonal.",
  },
  colon: {
    signatures: ["V = colon(A, B)", "V = colon(A, D, B)"],
    description: "Create vector A:B or A:D:B.",
  },
  magic: {
    signatures: ["M = magic(N)"],
    description: "N-by-N magic square. N must be >= 3.",
  },

  // ── Statistics ─────────────────────────────────────────────────────────
  corrcoef: {
    signatures: ["R = corrcoef(A)", "R = corrcoef(A, B)"],
    description: "Correlation coefficients.",
  },
  cov: {
    signatures: ["C = cov(A)", "C = cov(A, B)"],
    description: "Covariance matrix.",
  },
  accumarray: {
    signatures: [
      "A = accumarray(SUBS, VAL)",
      "A = accumarray(SUBS, VAL, SZ, FUN)",
    ],
    description: "Accumulate values into array using subscript indices.",
  },
};

for (const [name, help] of Object.entries(H)) {
  registerBuiltinHelp(name, help);
}
