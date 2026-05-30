% DIAGNOSIS: bridge-resolve.ts has an unconditional console.log (not gated by
% --verbose) that writes "[<op>] using bridge: native LAPACK addon" to STDOUT
% the first time a LAPACK-backed op resolves. Under --opt 0 a matmul resolves
% the bridge and prints this line; under --opt 1 the JIT uses its own naive
% matmul kernel, never resolves the bridge, and prints nothing -> stdout differs.
% (Result 19 is integer so there is no numeric divergence here, isolating the log.)
%
% --opt 0 stdout:
% [matmul] using bridge: native LAPACK addon
% 19
%
% --opt 1 stdout:
% 19
function r = mm(A,B)
  C = A*B;
  r = C(1,1);
end
A = [1 2; 3 4];
B = [5 6; 7 8];
for k=1:200
  v = mm(A,B);
end
disp(v)
