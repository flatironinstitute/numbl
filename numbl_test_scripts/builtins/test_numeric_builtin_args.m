% Numeric builtins (math, arithmetic, reductions, fft, prng, special
% functions, numerical): arity errors, argument-type errors, and inputs
% that used to be rejected or to lose their imaginary part.

% ── input / output count ────────────────────────────────────────────────
check_err('sin()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('sin(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('[a, b] = sin(1)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('hypot(3)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('hypot(1, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('round(1, 2, 3, 4, 5, 6)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('[a, b, c] = log2(8)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('[a, b, c] = max([1 2])', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('sum()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('sum(1, 2, 3, 4, 5)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('any(1, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('diff(1, 2, 3, 4)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('xor(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('atan2(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('[a, b] = mod(1, 2)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('fft()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('randperm(1, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('besselj(0)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('gallery()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('ellipj(0.5)', 'MATLAB:ellipj:NotEnoughInputs', 'Not enough input arguments.');
% M-file functions use different "too many" identifiers
check_err('sec(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b] = sec(1)', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('fftshift(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('rng(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('nthroot(8)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('conv(1)', 'MATLAB:minrhs', 'Not enough input arguments.');

assert(nargin(@sin) == 1);
assert(nargin(@hypot) == 2);
assert(nargin(@round) == 5);
assert(nargin(@atan2) == 2);
assert(nargin(@sec) == 1);
assert(nargin(@besselj) == 3);
assert(nargin(@fft) == 3);
assert(nargin(@ifft) == 4);
assert(nargin(@randperm) == 2);
assert(nargin(@diff) == 3);
assert(nargin(@any) == 2);
assert(nargin(@sum) == 4);
assert(nargin(@max) == 4);

% ── argument types ──────────────────────────────────────────────────────
float_msg = 'Invalid data type. Argument must be double or single.';
check_err('sin(''a'')', 'MATLAB:math:mustBeFloat', float_msg);
check_err('exp({})', 'MATLAB:math:mustBeFloat', float_msg);
check_err('hypot(''a'', 1)', 'MATLAB:math:mustBeFloat', float_msg);
check_err('erf(1i)', 'MATLAB:erf:notFullReal', 'Input must be real and full.');
check_err('gamma([1i 2])', 'MATLAB:gamma:notFullReal', 'Input must be real and full.');
check_err('realsqrt(1i)', 'MATLAB:validators:mustBeReal', ...
  'Invalid argument at position 1. Value must be real.');
check_err('realsqrt(-1)', 'MATLAB:realsqrt:complexResult', ...
  'Realsqrt produced complex result.');
check_err('factorial(''a'')', 'MATLAB:factorial:unsupportedType', 'Input must be numeric.');
check_err('atan2(1i, 1)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('atan2d(1i, 1)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('mod(5i, 2)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('rem([1i 2], 3)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('xor(1i, 0)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('nthroot(1i, 3)', 'MATLAB:nthroot:ComplexInput', 'Both X and N must be real.');
check_err('round(2.5, true)', 'MATLAB:round:SecondArg', ...
  'Second argument must be a real integer scalar.');
check_err('round(2.5, 1, ''foo'')', 'MATLAB:round:ThirdArg', ...
  'With 3 or 5 input arguments, the third input must be either ''decimals'' or ''significant''.');
check_err('round(''a'', 1)', 'MATLAB:round:NonfloatMultipleArgs', ...
  'First argument must be double or single with more than one input.');
check_err('round({})', 'MATLAB:round:FirstArg', ...
  'First argument must be a numeric, logical, or char array.');
check_err('min({})', 'MATLAB:min:wrongInput', 'First input array is an invalid data type.');
check_err('max(''ab'', ''b'')', 'MATLAB:max:wrongSecondInputText', ...
  'Invalid second argument. To specify an additional argument, use max(A,[],...).');
check_err('max(1, 2, 3)', 'MATLAB:max:caseNotSupported', ...
  'Specifying the operating dimension is not supported for two input arrays.');
check_err('[a, b] = min([1 2], [3 0])', 'MATLAB:min:TwoInTwoOutCaseNotSupported', ...
  'Two output arguments are not supported for two input arrays.');
check_err('sum({})', 'MATLAB:sum:wrongInput', ...
  'Invalid data type. First argument must be numeric or logical.');
check_err('mean({})', 'MATLAB:sum:wrongInput', ...
  'Invalid data type. First argument must be numeric or logical.');
check_err('any({})', 'MATLAB:any:wrongInput', ...
  'Invalid data type. First argument must be numeric or logical.');
check_err('prod(''ab'')', 'MATLAB:prod:charInput', 'PROD does not work with char inputs.');
check_err('cumsum(''ab'')', 'MATLAB:cumsum:wrongInput', ...
  'Invalid data type. First input argument must be numeric or logical.');
check_err('var([1 2 3], 2)', 'MATLAB:var:invalidWgts', ...
  'W must be a double or single vector of nonnegative weights, or a scalar 0 or 1.');
check_err('diff([1 2 3], 1i)', 'MATLAB:diff:differenceOrderMustBePositiveInteger', ...
  'Difference order N must be a positive integer scalar.');
check_err('sum([1 2], 1i)', 'MATLAB:getdimarg:invalidDim', ...
  'Dimension argument must be a positive integer scalar, a vector of unique positive integers, or ''all''.');
check_err('fft(''ab'')', 'MATLAB:fftfcn:InvalidInputType', ...
  'Invalid data type. First argument must be double, single, int8, uint8, int16, uint16, int32, uint32, or logical.');
check_err('fft(sparse([1 2]))', 'MATLAB:fftfcn:SparseInputType', ...
  'Sparse matrices are not supported.');
check_err('fft([1 2], 1i)', 'MATLAB:fftfcn:lengthNotNonNegInt', ...
  'FFT length must be a nonnegative integer scalar.');
check_err('fftshift([1 2], 1i)', 'MATLAB:fftshift:DimNotPosInt', 'DIM must be a positive integer.');
check_err('rand(2+1i)', 'MATLAB:obsoleteComplexInput', 'Size inputs must be real.');
check_err('rand({})', 'MATLAB:rand:inputType', 'Size inputs must be numeric.');
check_err('rand(''a'')', 'MATLAB:rand:invalidOutputType', ...
  'Class name must be a class that supports RAND, such as "double" or "single".');
check_err('randi([1+1i 3])', 'MATLAB:randi:invalidLimits', ...
  'First input must be a positive scalar integer value IMAX, or two integer values [IMIN IMAX] with IMIN less than or equal to IMAX.');
check_err('randperm(3, 5)', 'MATLAB:randperm:inputKTooLarge', 'K must be less than or equal to N.');
check_err('rng(1i)', 'MATLAB:rng:badSeed', 'SD must be a nonnegative integer seed less than 2^32.');
check_err('rng(''foo'')', 'MATLAB:rng:unknownRNGType', '''foo'' is not a valid generator type.');
check_err('eps(1, 2)', 'MATLAB:eps:mustBeLike', ...
  'With two input arguments, the first argument must be "like".');
check_err('eps(''foo'')', 'MATLAB:eps:invalidClassName', ...
  'Class name must be a class that supports EPS, such as "double" or "single".');
check_err('besselj(1i, 2)', 'MATLAB:besselj:nonRealNU', 'NU must be real.');
check_err('besselj([0 1], [1 2 3])', 'MATLAB:besselj:NUAndZSizeMismatch', ...
  'NU and Z must be the same size or one must be a scalar.');
check_err('besselh(0, 3, 2)', 'MATLAB:besselh:KNotOneOrTwo', 'K must be 1 or 2.');
check_err('airy(5, 1)', 'MATLAB:airy:InvalidOrder', 'K must be 0, 1, 2, or 3.');
check_err('ellipj(1i, 0.3)', 'MATLAB:ellipj:ComplexInputs', 'Input arguments must be real.');
check_err('legendre(1i, 0.5)', 'MATLAB:legendre:InvalidN', 'N must be a positive scalar integer.');
check_err('legendre(2, [0.1 1i])', 'MATLAB:legendre:InvalidX', ...
  'X must be real and in the range (-1,1).');
check_err('interp1([1 2], [1 2], 1i)', 'MATLAB:griddedInterpolant:ComplexDataPointErrId', ...
  'Input coordinates must be real.');
check_err('ppval(mkpp([0 1], [1 2]), 1i)', 'MATLAB:chckxy:XQComplex', ...
  'The interpolation points must be real.');
check_err('discretize([1i 2], [0 1 2 3])', 'MATLAB:discretize:invalidType', ...
  'Invalid data type. Data to be binned must be a real numeric or logical array.');

% ── complex input (no longer dropped or rejected) ──────────────────────
tol = 1e-5;
assert(hypot(3i, 4) == 5);
assert(isequal(hypot([3i 1], 4), [5 sqrt(17)]));
assert(abs(sec(0.5+0.5i) - (0.949979+0.239828i)) < tol);
assert(abs(csc(0.5+0.5i) - (1.07823-0.912074i)) < tol);
assert(abs(cot(0.5+0.5i) - (0.83914-1.17195i)) < tol);
assert(abs(sech(0.5+0.5i) - (0.949979-0.239828i)) < tol);
assert(abs(coth(0.5+0.5i) - (1.17195-0.83914i)) < tol);
assert(abs(acot(0.5+0.5i) - (1.01722-0.402359i)) < tol);
assert(abs(sind(0.5+0.5i) - (0.00872687+0.00872642i)) < 1e-8);
assert(abs(atand(0.5+0.5i) - (31.7175+23.0535i)) < 1e-4);
assert(abs(pow2(0.5+0.5i) - (1.33013+0.480376i)) < tol);
v = sec([0.5+0.5i 0.25]);
assert(abs(v(2) - 1.03209) < tol);
assert(nextpow2(0.5+0.5i) == 0);
assert(isequal(nextpow2([-8 0.25 0 5]), [3 -2 0 3]));
assert(abs(eps(1i) - eps(1)) == 0);
assert(isequal(eps([1 1i]), [eps eps]));
assert(isequal(conv([1 1i], [1 1]), [1, 1+1i, 1i]));
assert(isequal(conv([1 2], 1i), [1i 2i]));
assert(polyval([1 1i], 2) == 2+1i);
assert(polyval([1 2], 1i) == 2+1i);
assert(isequal(polyval([1 2], [1i 2]), [2+1i 4]));
assert(isequal(poly(1i), [1 -1i]));
assert(isequal(poly([1i 2]), [1, -2-1i, 2i]));
assert(prod([1i 2], 'all') == 2i);
assert(abs(var([1+1i 2 3-2i]) - 10/3) < 1e-12);
assert(abs(std([1i 2]) - sqrt(2.5)) < 1e-12);
assert(max(abs(var([1+1i 2; 3-2i 4], 0, 2) - [1; 2.5])) < 1e-12);
assert(isequal(cummax([1i 2 -3]), [1i 2 -3]));
assert(cumsum(1i) == 1i);
assert(cumprod(2i) == 2i);
assert(isempty(diff(1i)));
assert(power(2, 3i) == 2^(3i));
assert(abs(airy(0, 1i, 1) - (0.274305-0.0825607i)) < tol);

% ── char and logical input ──────────────────────────────────────────────
assert(min('abc') == 97);
[m, k] = max('abc');
assert(m == 99 && k == 3);
assert(mod('a', 3) == 1);
assert(rem('a', 3) == 1);
assert(power('a', 2) == 9409);
assert(sum('abc') == 294);
assert(mean('ab') == 97.5);
assert(abs(var('abc') - 1) < 1e-12);
assert(any('a'));
assert(all(''));
assert(isequal(diff('abc'), [1 1]));
assert(xor('a', 0));
assert(nthroot('a', 2) == sqrt(97));
assert(trapz('abc') == 196);
assert(cov('abc') == 1);
assert(isequal(histc('ab', [96 97 98 99]), [0 1 1 0]));
assert(isequal(gradient('abc'), [1 1 1]));
assert(polyval([1 2], 'a') == 99);
c = corrcoef('abc', [1 2 4]);
assert(abs(c(1, 2) - 0.981981) < 1e-6);
assert(std(true) == 0);
assert(cumsum(true) == 1);
assert(cumprod(true) == 1);
assert(isempty(diff(true)));
assert(fft(true) == 1);
assert(ifft(true) == 1);
assert(islogical(fftshift(true)));
assert(islogical(fftshift([true false])));
assert(islogical(cummax(true)));
assert(islogical(cummax([true false true])));
assert(polyval([1 2], true) == 3);
assert(isequal(accumarray([1; 2], true), [1; 1]));

% ── other forms ─────────────────────────────────────────────────────────
assert(isequal(atan2d([1 1], [1 -1]), [45 135]));
assert(pow2(3, 2) == 12);
assert(isequal(pow2([1 2], 3), [8 16]));
assert(round(1234.5, 2, 'significant') == 1200);
assert(abs(round(2.567, 1, 'decimals') - 2.6) < 1e-12);
[f, e] = log2(8);
assert(f == 0.5 && e == 4);
assert(max([1 5 3], [], 2, 'omitnan') == 5);
assert(min([1 NaN 3], [], 'omitnan') == 1);
[s, mu] = std([1 2 3]);
assert(s == 1 && mu == 2);
[s, mu] = var([1 2; 3 5], 0, 2);
assert(isequal(s, [0.5; 2]) && isequal(mu, [1.5; 4]));
assert(eps('like', 1) == eps);
assert(abs(besselj(0, 2, 1) - besselj(0, 2)) < 1e-15);
assert(isequal(size(randi(10, 2, 3, 'int32')), [2 3]));

disp('SUCCESS')

function check_err(stmt, id, msg)
  ok = false;
  try
    eval([stmt ';']);
  catch e
    ok = true;
    assert(strcmp(e.identifier, id), ...
      sprintf('%s: identifier ''%s'', expected ''%s''', stmt, e.identifier, id));
    assert(strcmp(e.message, msg), ...
      sprintf('%s: message ''%s'', expected ''%s''', stmt, e.message, msg));
  end
  assert(ok, sprintf('%s: expected an error', stmt));
end
