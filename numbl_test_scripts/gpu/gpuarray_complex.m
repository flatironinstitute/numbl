% Complex gpuArrays and the FFT agree with the same operations on the host,
% within single-precision rounding. As in MATLAB, a complex result on the
% GPU stays complex when its imaginary parts are zero, fft always returns
% complex data, and ifft does unless 'symmetric' is given.

rng(3);
A = single(randn(7, 9) + 1i * randn(7, 9));
B = single(randn(7, 9) + 1i * randn(7, 9));
R = single(randn(7, 9));
GA = gpuArray(A);
GB = gpuArray(B);
GR = gpuArray(R);

% arithmetic, with complex and real operands
same(GA + GB, A + B, 0);
same(GA - GR, A - R, 0);
same(GA .* GB, A .* B, 1e-6);
same(GA .* GR, A .* R, 0);
same(GA ./ GB, A ./ B, 1e-6);
same(GA .\ GB, A .\ B, 1e-6);
same(GA ./ 2, A ./ 2, 0);
same(-GA, -A, 0);
same(GA * 2i, A * 2i, 0);
same(GR + 1i, R + 1i, 0);
same(GA .^ 2, A .^ 2, 1e-6);
same(GA + gpuArray(B(1, :)), A + B(1, :), 0);

% functions of complex data (exp, sin and cos of complex data use WGSL's
% sin and cos, accurate to an absolute 2^-11 on [-pi, pi])
same(exp(GA), exp(A), 1e-3);
same(log(GA), log(A), 1e-5);
same(sqrt(GA), sqrt(A), 1e-6);
same(sin(GA), sin(A), 1e-3);
same(cos(GA), cos(A), 1e-3);
same(abs(GA), abs(A), 1e-6);
same(angle(GA), angle(A), 1e-5);
same(conj(GA), conj(A), 0);
same(real(GA), real(A), 0);
same(imag(GA), imag(A), 0);
same(GA == GA, A == A, 0);
same(GA ~= GB, A ~= B, 0);
N = A;
N(3) = complex(NaN, 1);
N(4) = complex(1, Inf);
same(isnan(gpuArray(N)), isnan(N), 0);
same(isinf(gpuArray(N)), isinf(N), 0);
same(isfinite(gpuArray(N)), isfinite(N), 0);

% making complex data, and complex results of real data
same(complex(GR, GR), complex(R, R), 0);
same(sqrt(complex(gpuArray(single([-4 4])))), complex(single([0 2]), single([2 0])), 0);
same(log(complex(-GR)), log(complex(-R)), 1e-5);
same(imag(GR), zeros(7, 9, 'single'), 0);
same(angle(GR), angle(R), 1e-6);

% products, transposes, reductions
C = single(randn(9, 4) + 1i * randn(9, 4));
same(GA * gpuArray(C), A * C, 1e-5);
same(GR * gpuArray(C), R * C, 1e-5);
same(GA * gpuArray(real(C)), A * real(C), 1e-5);
same(GA', A', 0);
same(GA.', A.', 0);
same(GA(:, 1)', A(:, 1)', 0);
same(sum(GA), sum(A), 1e-5);
same(sum(GA, 2), sum(A, 2), 1e-5);
same(mean(GA, 'all'), mean(A, 'all'), 1e-5);
same(cumsum(GA), cumsum(A), 1e-5);
same(dot(GA(:, 1), GB(:, 2)), dot(A(:, 1), B(:, 2)), 1e-5);
same(norm(GA(:, 1)), norm(A(:, 1)), 1e-5);

% indexing, assignment, concatenation, repmat
same(GA(2, :), A(2, :), 0);
same(GA(GR > 0), A(R > 0), 0);
G2 = GA;
G2(2, 2) = 5;
H2 = A;
H2(2, 2) = 5;
same(G2, H2, 0);
G2(:, 1) = GB(:, 1);
H2(:, 1) = B(:, 1);
same(G2, H2, 0);
G3 = GR;
G3(1) = 2i;
H3 = R;
H3(1) = 2i;
same(G3, H3, 0);
same([GA; GB], [A; B], 0);
same([GA, GR], [A, R], 0);
same(repmat(GA(1:2, 1:3), 2, 2), repmat(A(1:2, 1:3), 2, 2), 0);

% zero imaginary parts stay
assert(~isreal(GA + 1i - 1i));
assert(~isreal(GA(1) - 1i * imag(GA(1))));
G4 = gpuArray(single([1+2i 3]));
assert(~isreal(G4(2)));
assert(~isreal(sum(gpuArray(single([1i -1i])))));
assert(~isreal(complex(gpuArray(single(1)))));
assert(isreal(real(GA)) && isreal(abs(GA)));

% fft and ifft: power-of-2 lengths and others (Bluestein's algorithm)
for n = [1 2 3 5 8 12 64 100 257 1024]
  X = single(randn(n, 3) + 1i * randn(n, 3));
  Xr = real(X);
  same(fft(gpuArray(X)), fft(X), 1e-5);
  same(ifft(gpuArray(X)), ifft(X), 1e-5);
  same(fft(gpuArray(Xr)), fft(Xr), 1e-5);
  same(fft(gpuArray(Xr'), [], 2), fft(Xr', [], 2), 1e-5);
  same(ifft(gpuArray(X), [], 1), ifft(X, [], 1), 1e-5);
end
X = single(randn(6, 5));
same(fft(gpuArray(X), 8), fft(X, 8), 1e-5);
same(fft(gpuArray(X), 4), fft(X, 4), 1e-5);
same(fft(gpuArray(X), 7, 2), fft(X, 7, 2), 1e-5);
same(ifft(fft(gpuArray(X))), complex(X), 1e-5);
X3 = single(randn(4, 3, 5));
same(fft(gpuArray(X3), [], 3), fft(X3, [], 3), 1e-5);
assert(~isreal(fft(gpuArray(single(5)))));
assert(~isreal(ifft(fft(gpuArray(X)))));

% ifft(..., 'symmetric') is real
Y = fft(single(randn(10, 2)));
Y(8, 1) = 100;
same(ifft(gpuArray(Y), 'symmetric'), ifft(Y, 'symmetric'), 1e-5);
same(ifft(gpuArray(Y), 16, 'symmetric'), ifft(Y, 16, 'symmetric'), 1e-5);
same(ifft(gpuArray(Y), [], 1, 'nonsymmetric'), ifft(Y, [], 1), 1e-5);
assert(isreal(ifft(gpuArray(Y), 'symmetric')));

disp('SUCCESS')

function same(g, h, tol)
  assert(isgpuarray(g), 'result is not a gpuArray');
  assert(strcmp(underlyingType(g), class(h)), sprintf('class %s, expected %s', underlyingType(g), class(h)));
  x = gather(g);
  assert(isequal(size(x), size(h)), 'size differs');
  % a complex host result is complex on the GPU too (the converse need
  % not hold: the host drops zero imaginary parts)
  assert(isreal(h) || ~isreal(x), 'result is real, expected complex');
  d = double(x(:));
  e = double(h(:));
  nan_ok = isequal(isnan(d), isnan(e));
  d(isnan(d)) = 0;
  e(isnan(e)) = 0;
  err = max([0; abs(d - e)]) / max([1; abs(e)]);
  assert(nan_ok && err <= tol, sprintf('error %g exceeds %g', err, tol));
end
