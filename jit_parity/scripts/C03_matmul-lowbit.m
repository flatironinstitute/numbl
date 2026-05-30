% DIAGNOSIS: A*B (matrix multiply) diverges in low-order bits. The interpreter
% uses the native LAPACK addon's BLAS dgemm (blocked/SIMD accumulation); the
% JS-JIT emits a naive triple-loop matmul. Different accumulation order ->
% different floating-point rounding. (Same native-vs-JIT class as sum/exp.)
% NOTE: opt0 also prints a spurious "[matmul] using bridge" line (see finding 05);
% it is stripped here via stderr/2>/dev/null when comparing the numbers.
%
% --opt 0 output (number only):
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
