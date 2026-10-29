% Builtins with single arguments: which outputs are single, the class
% arguments ('single', 'like') of the constructors, and the precision
% queries. Verified against MATLAB R2025b.

A = single([1.5 2.25; 3.5 4.75]);
v = single([0.5 0.25 0.75]);
s = single(0.5);

% Element-wise math, rounding and complex parts keep the class
cases = {
  'abs(-v)', 'single'; 'sqrt(v)', 'single'; 'exp(v)', 'single'; 'log(v)', 'single';
  'sin(v)', 'single'; 'cosd(v)', 'single'; 'atan2(v, 1)', 'single'; 'hypot(v, v)', 'single';
  'floor(v)', 'single'; 'round(v)', 'single'; 'fix(v)', 'single'; 'mod(v, 0.3)', 'single';
  'rem(v, 0.3)', 'single'; 'sign(v)', 'single'; 'real(v)', 'single'; 'imag(v)', 'single';
  'conj(v)', 'single'; 'angle(v)', 'single'; 'complex(v, 1)', 'single'; 'gamma(v)', 'single';
  'erf(v)', 'single'; 'besselj(0, v)', 'single'; 'nthroot(v, 3)', 'single'; 'power(v, 2)', 'single';
  'plus(v, 1)', 'single'; 'times(2, v)', 'single'; 'mtimes(A, A)', 'single'; 'mldivide(A, [1; 2])', 'single';
  'uminus(v)', 'single'; 'eq(v, 1)', 'logical'; 'isnan(v)', 'logical'; 'isfinite(v)', 'logical';
  % reductions and cumulative
  'sum(v)', 'single'; 'prod(v)', 'single'; 'mean(v)', 'single'; 'median(v)', 'single';
  'std(v)', 'single'; 'var(v)', 'single'; 'cumsum(v)', 'single'; 'cumprod(v)', 'single';
  'max(v)', 'single'; 'min(A, [], 2)', 'single'; 'max(v, 0.4)', 'single'; 'diff(v)', 'single';
  'norm(v)', 'single'; 'dot(v, v)', 'single'; 'any(v)', 'logical'; 'sum(v, ''double'')', 'double';
  'sum(A, 2, ''double'')', 'double'; 'prod(v, ''double'')', 'double'; 'mean(v, ''double'')', 'double';
  'mean(v, ''native'')', 'single'; 'sum(v, ''native'')', 'single';
  % linear algebra and signal processing
  'inv(A)', 'single'; 'det(A)', 'single'; 'trace(A)', 'single'; 'pinv(A)', 'single';
  'expm(A)', 'single'; 'kron(A, A)', 'single'; 'conv(v, v)', 'single'; 'fft(v)', 'single';
  'ifft(v)', 'single'; 'polyval(v, 2)', 'single'; 'cross(v, [1 2 3])', 'single';
  % shape and set operations
  'reshape(A, 1, 4)', 'single'; 'repmat(v, 2, 1)', 'single'; 'flip(v)', 'single';
  'circshift(v, 1)', 'single'; 'permute(A, [2 1])', 'single'; 'tril(A)', 'single';
  'sort(v)', 'single'; 'unique(v)', 'single'; 'cat(1, v, v)', 'single'; 'cat(3, A, A)', 'single'; 'cat(single(2), 1, 2)', 'double'; 'horzcat(v, 1)', 'single';
  'diag(A)', 'single'; 'nonzeros(v)', 'single'; 'linspace(s, 1, 3)', 'single'; 'colon(s, 2)', 'single';
  % index and size outputs stay double
  'size(A)', 'double'; 'numel(A)', 'double'; 'find(v)', 'double'; 'length(v)', 'double';
  'nnz(v)', 'double'; 'ndims(A)', 'double';
  % constructors
  'zeros(2, ''single'')', 'single'; 'ones(1, 3, ''single'')', 'single'; 'eye(2, ''single'')', 'single';
  'nan(2, ''single'')', 'single'; 'inf(1, ''single'')', 'single'; 'zeros(2, ''like'', s)', 'single';
  'ones(2, ''like'', 1)', 'double'; 'rand(2, ''single'')', 'single'; 'randn(1, 2, ''single'')', 'single';
  'randi(5, 2, ''single'')', 'single'; 'rand(2, ''like'', s)', 'single'; 'zeros(2, ''double'')', 'double';
  'zeros(single(2))', 'double';
  % conversion
  'cast(1, ''single'')', 'single'; 'cast(s, ''double'')', 'double'; 'cast(1, ''like'', s)', 'single';
  'cast(s, ''like'', 2)', 'double'; 'logical(v)', 'logical'; 'double(s)', 'double'};
for k = 1:size(cases, 1)
  got = class(eval(cases{k, 1}));
  assert(strcmp(got, cases{k, 2}), sprintf('%s: got %s, expected %s', cases{k, 1}, got, cases{k, 2}));
end

% Index outputs of multi-output builtins stay double
[m, i] = max(v);
assert(strcmp(class(m), 'single') && strcmp(class(i), 'double') && i == 3);
[m, i] = min(A, [], 2);
assert(strcmp(class(m), 'single') && strcmp(class(i), 'double'));
[sv, si] = sort(v);
assert(strcmp(class(sv), 'single') && strcmp(class(si), 'double'));
[r, c, val] = find(single([0 2; 3 0]));
assert(strcmp(class(r), 'double') && strcmp(class(c), 'double') && strcmp(class(val), 'single'));
[u, ia, ic] = unique(single([3 1 3]));
assert(strcmp(class(u), 'single') && strcmp(class(ia), 'double') && strcmp(class(ic), 'double'));
[q, rr] = qr(A);
assert(strcmp(class(q), 'single') && strcmp(class(rr), 'single'));
[V, D] = eig(A);
assert(strcmp(class(V), 'single') && strcmp(class(D), 'single'));
[X, Y] = meshgrid(v, 1:2);
assert(strcmp(class(X), 'single') && strcmp(class(Y), 'double'));

% Results are rounded to single precision
assert(double(sqrt(single(2))) == 1.4142135381698608);
assert(double(sum(single([0.1 0.2]))) == double(single(0.1) + single(0.2)));
assert(double(mean(single([1 2 4]))) == double(single(7/3)));
assert(double(exp(single(1))) == double(single(exp(1))));

% Precision queries
assert(strcmp(class(eps('single')), 'single') && double(eps('single')) == 2^-23);
assert(double(eps(single(1))) == 2^-23 && double(eps(single(1000))) == 2^-14);
assert(double(eps(single(0))) == 2^-149);
assert(isequal(double(eps(single([1 2 4]))), 2.^[-23 -22 -21]));
assert(double(eps('like', s)) == 2^-23 && eps('like', 1) == 2^-52);
assert(eps('double') == 2^-52 && eps(1) == 2^-52);
assert(strcmp(class(realmax('single')), 'single') && double(realmax('single')) == 3.4028234663852886e38);
assert(double(realmin('single')) == 2^-126 && realmin('double') == realmin);
assert(double(flintmax('single')) == 2^24 && flintmax('double') == 2^53);
assert(realmax == realmax('double') && double(realmax('like', s)) == double(realmax('single')));

% typecast reads and writes 4 bytes per single element
assert(double(typecast(single(1), 'uint32')) == 1065353216);
assert(isequal(double(typecast(single([1 2]), 'uint8')), [0 0 128 63 0 0 0 64]));
assert(strcmp(class(typecast(single([1 2]), 'single')), 'single'));

% Constructor values
z = zeros(2, 3, 'single');
assert(isequal(size(z), [2 3]) && all(z(:) == 0));
r = rand(3, 'single');
assert(all(double(r(:)) == double(single(r(:)))) && all(r(:) >= 0 & r(:) < 1));
w = whos('z');
assert(w.bytes == 24);

disp('SUCCESS')
