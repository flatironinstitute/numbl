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

% functions with a real domain, inside it
P = gpuArray(abs(A) + single(0.1));
Ph = abs(A) + single(0.1);
same(sqrt(P), sqrt(Ph), 1e-6);
same(log(P), log(Ph), 1e-6);
same(log2(P), log2(Ph), 1e-6);
same(log10(P), log10(Ph), 1e-6);
same(log1p(P), log1p(Ph), 1e-6);
same(P .^ 2, Ph .^ 2, 0);
same(P .^ 0.5, Ph .^ 0.5, 1e-6);
same((-P) .^ 3, (-Ph) .^ 3, 1e-6);
same(P .^ 0, Ph .^ 0, 0);
same(asin(P / 4), asin(Ph / 4), 1e-3);
same(acos(P / 4), acos(Ph / 4), 1e-3);
same(atanh(P / 4), atanh(Ph / 4), 1e-3);
same(acosh(P + 1), acosh(Ph + 1), 1e-5);

% outside the domain: an error, as on every device
for f = {@() sqrt(-P), @() log(-P), @() (-P) .^ 0.5, @() acosh(P - 2)}
  try
    f{1}();
    error('expected an error');
  catch e
    assert(contains(e.message, 'needs to return a complex result'), e.message);
  end
end

% implicit expansion
r = single(randn(1, 53));
c = single(randn(37, 1));
same(GA + gpuArray(r), A + r, 0);
same(GA .* gpuArray(c), A .* c, 0);
same(gpuArray(c) - gpuArray(r), c - r, 0);
same(gpuArray(c) < gpuArray(r), c < r, 0);

% indexing and assignment
same(GA(:, 2), A(:, 2), 0);
same(GA(3, :), A(3, :), 0);
same(GA(GA > 0.5), A(A > 0.5), 0);
same(GA([1 3], [2 4]), A([1 3], [2 4]), 0);
same(GA(:), A(:), 0);
same(GA(end), A(end), 0);
same(GA(end:-1:1, 1), A(end:-1:1, 1), 0);
G2 = GA;
G2(2, :) = 0;
H2 = A;
H2(2, :) = 0;
same(G2, H2, 0);
G2(G2 > 0.5) = -1;
H2(H2 > 0.5) = -1;
same(G2, H2, 0);
G2(:, 1) = gpuArray(single(1:37)');
H2(:, 1) = single(1:37)';
same(G2, H2, 0);
same(GA, A, 0);

% predicates, mod/rem, atan2/hypot
N = A;
N(5) = NaN;
N(9) = Inf;
same(isnan(gpuArray(N)), isnan(N), 0);
same(isinf(gpuArray(N)), isinf(N), 0);
same(isfinite(gpuArray(N)), isfinite(N), 0);
same(mod(GA * 3, 0.7), mod(A * 3, 0.7), 1e-6);
same(rem(GA * 3, 0.7), rem(A * 3, 0.7), 1e-6);
same(mod(gpuArray(single([5 -5 0])), 0), single([5 -5 0]), 0);
same(rem(gpuArray(single([5 -5])), 0), single([NaN NaN]), 0);
same(atan2(GA, GB), atan2(A, B), 1e-5);
same(hypot(GA, GB), hypot(A, B), 1e-6);

% any/all, index outputs, cumulative sums, dot, norm
same(any(GA > 2), any(A > 2), 0);
same(all(GA > -2, 2), all(A > -2, 2), 0);
same(any(GA > 3, 'all'), any(A > 3, 'all'), 0);
[m, i] = max(gpuArray(N));
[mh, ih] = max(N);
same(m, mh, 0);
same(i, ih, 0);
[m, i] = min(GB, [], 2);
[mh, ih] = min(B, [], 2);
same(m, mh, 0);
same(i, ih, 0);
[m, i] = max(gpuArray(v));
[mh, ih] = max(v);
same(m, mh, 0);
same(i, ih, 0);
same(cumsum(GA), cumsum(A), 1e-5);
same(cumsum(GA, 2), cumsum(A, 2), 1e-5);
same(cumprod(GA(1:6, :) / 2), cumprod(A(1:6, :) / 2), 1e-5);
same(cumsum(gpuArray(v)), cumsum(v), 1e-5);
same(dot(gpuArray(v), gpuArray(v)), dot(v, v), 1e-5);
same(norm(gpuArray(v)), norm(v), 1e-5);
same(norm(gpuArray(v), 1), norm(v, 1), 1e-5);
same(norm(gpuArray(v), Inf), norm(v, Inf), 0);

% concatenation and repmat
same([GA, GB], [A, B], 0);
same([GA; GB], [A; B], 0);
same(cat(3, GA, GB), cat(3, A, B), 0);
same([GA(:, 1), ones(37, 1)], [A(:, 1), ones(37, 1, 'single')], 0);
same([gpuArray(single(1)), 2, 3], single([1 2 3]), 0);
same(repmat(GA(1:3, 1:2), 2, 3), repmat(A(1:3, 1:2), 2, 3), 0);
same(repmat(GA(1, :), [4 1]), repmat(A(1, :), [4 1]), 0);

% device memory is released when arrays go away (AvailableMemory)
d = gpuDevice;
used0 = d.TotalMemory - d.AvailableMemory;
X = gpuArray(single(rand(500)));
for k = 1:50
  X = X .* 0.5 + sum(X(:)) / 1e6;
end
Y = reshape(X, 250, []);
c = {X, Y'};
clear X Y c
d = gpuDevice;
assert(d.TotalMemory - d.AvailableMemory <= used0, 'device memory was not released');

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
