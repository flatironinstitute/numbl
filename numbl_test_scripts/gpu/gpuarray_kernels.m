% Operations on single gpuArrays agree with the same operations on the host,
% within the accuracy GPU arithmetic allows (WGSL division is accurate to
% 2.5 ulp, sin and cos to an absolute 2^-11 on [-pi, pi]). On numbl's CPU
% reference devices the results are exact.

rng(1);
A = single(randn(37, 53));
B = single(randn(37, 53));
C = single(randn(53, 29));
GA = gpuArray(A);
GB = gpuArray(B);
GC = gpuArray(C);

% exact: +, -, *, comparisons, rounding, abs, transpose
same(GA + GB, A + B, 0);
same(GA - 2.5, A - 2.5, 0);
same(2 - GA, 2 - A, 0);
same(GA .* GB, A .* B, 0);
same(-GA, -A, 0);
same(GA < GB, A < B, 0);
same(GA >= 0, A >= 0, 0);
same(GA == GA, A == A, 0);
same(GA ~= 1, A ~= 1, 0);
same((GA > 0) & (GB > 0), (A > 0) & (B > 0), 0);
same((GA > 0) | (GB > 0), (A > 0) | (B > 0), 0);
same(~(GA > 0), ~(A > 0), 0);
same(abs(GA), abs(A), 0);
same(floor(GA * 3), floor(A * 3), 0);
same(ceil(GA * 3), ceil(A * 3), 0);
same(fix(GA * 3), fix(A * 3), 0);
same(round(GA * 3), round(A * 3), 0);
same(round(gpuArray(single([0.5 -0.5 1.5 2.5 -2.5 0.49999997]))), single([1 -1 2 3 -3 0]), 0);
same(sign(GA), sign(A), 0);
same(GA', A', 0);
same(GA.', A.', 0);
same(max(GA, GB), max(A, B), 0);
same(min(GA, 0), min(A, 0), 0);
same(max(GA), max(A), 0);
same(min(GA, [], 2), min(A, [], 2), 0);
same(max(gpuArray(single([NaN 1 2 NaN]))), single(2), 0);
same(min(gpuArray(single([NaN NaN]))), single(NaN), 0);

% within a few ulp: division, transcendental functions
same(GA ./ GB, A ./ B, 1e-6);
same(3 .\ GA, 3 .\ A, 1e-6);
same(exp(GA), exp(A), 1e-6);
same(tanh(GA), tanh(A), 1e-6);
same(sinh(GA), sinh(A), 1e-5);
same(cosh(GA), cosh(A), 1e-5);
same(sin(GA), sin(A), 1e-3);
same(cos(GA), cos(A), 1e-3);
same(tan(GA / 4), tan(A / 4), 1e-3);

% reductions and products (summation order differs)
same(sum(GA), sum(A), 1e-5);
same(sum(GA, 2), sum(A, 2), 1e-5);
same(sum(GA, 'all'), sum(A, 'all'), 1e-5);
same(mean(GA), mean(A), 1e-5);
same(mean(GA, 2), mean(A, 2), 1e-5);
same(prod(GA(1:5, :)), prod(A(1:5, :)), 1e-5);
v = single(rand(1, 100000));
same(sum(gpuArray(v)), sum(v), 1e-5);
same(GA * GC, A * C, 1e-5);
same(GA' * GA, A' * A, 1e-5);

% classes and shapes of results
assert(strcmp(underlyingType(GA + 1), 'single') && strcmp(underlyingType(GA < 1), 'logical'));
assert(isequal(size(sum(GA)), [1 53]) && isequal(size(sum(GA, 2)), [37 1]));
assert(isequal(size(GA * GC), [37 29]) && isequal(size(GA'), [53 37]));
assert(strcmp(underlyingType(max(GA > 0)), 'logical'));

disp('SUCCESS')

function same(g, h, tol)
  assert(isgpuarray(g), 'result is not a gpuArray');
  assert(strcmp(underlyingType(g), class(h)), sprintf('class %s, expected %s', underlyingType(g), class(h)));
  x = gather(g);
  assert(isequal(size(x), size(h)), 'size differs');
  d = double(x(:));
  e = double(h(:));
  nan_ok = isequal(isnan(d), isnan(e));
  d(isnan(d)) = 0;
  e(isnan(e)) = 0;
  err = max([0; abs(d - e)]) / max([1; abs(e)]);
  assert(nan_ok && err <= tol, sprintf('error %g exceeds %g', err, tol));
end
