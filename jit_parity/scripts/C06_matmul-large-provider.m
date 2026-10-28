% Products above the inline threshold (MATMUL_INLINE_MAX_WORK multiply-adds)
% go through the LAPACK provider stack at every --opt level (BLAS when the
% native addon is loaded): the JS-JIT through its `$numeric` host hook and
% the C-JIT through a host callback. Before that the JITs ran a naive loop,
% so with the addon loaded every line below differed from --opt 0 in the
% last bits. Covers matrix * matrix, the 1xk * kx1 scalar form, and complex.
function [C, s, W] = f(A, B, x, Z)
  C = A * B;
  s = x' * x;
  W = Z * B;
end
A = reshape(1 ./ (1:1200), 40, 30);
B = reshape(1 ./ (3:1502), 30, 50);
x = 1 ./ (7:2006)';
Z = A(1:20, :) + 1i * A(21:40, :);
[C, s, W] = f(A, B, x, Z);
fprintf('%.17g\n', C(1,1), C(2,7), C(17,23), C(33,41), C(40,50), s);
fprintf('%.17g %.17g\n', real(W(1,1)), imag(W(1,1)), real(W(13,37)), imag(W(13,37)));
