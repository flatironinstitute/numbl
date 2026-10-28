% DIAGNOSIS: A*B (matrix multiply) diverged in low-order bits. The interpreter
% used the native LAPACK addon's BLAS dgemm (blocked/SIMD accumulation); the
% JS-JIT emitted a naive triple-loop matmul. Different accumulation order ->
% different floating-point rounding.
% FIX: an 8x8 product is below the inline threshold, so every mode now runs
% the same reference loop (the interpreter through the inline provider).
%
% --opt 0 output (before the fix):
% 0.57277808314492906
%
% --opt 1 output:
% 0.57277808314492917
function r = f(A, B)
  C = A * B;
  r = C(1,1);
end
A = reshape(1 ./ (1:64), 8, 8);
B = reshape(1 ./ (2:65), 8, 8);
for k=1:200
  a = f(A, B);
end
fprintf('%.17g\n', a);
