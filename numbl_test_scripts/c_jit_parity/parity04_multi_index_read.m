% C-JIT parity gap #04: multi-index (2D/3D) tensor Index read.
%
% The JS-JIT compiles `A(i,j)` / `B(i,j,k)` via idx2r_h / idx3r_h; the
% C-JIT historically bailed feasibility with
%   "multi-index Index read not supported"
% because multi-index reads need the tensor's shape dims (d0, d1)
% plumbed through the C ABI, and numbl_jit_runtime only had numbl_idx1r.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 45\n234
%   numbl --opt 2 run <this>                         -> 45\n234  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 45\n234  (the gap is now closed)
%   matlab -batch parity04_multi_index_read          -> 45\n234

A = [1 2 3; 4 5 6; 7 8 9];
s = sum_2d(A);
disp(s)

B = zeros(2, 3, 4);
for i = 1:2
    for j = 1:3
        for k = 1:4
            B(i,j,k) = i * 100 + j * 10 + k;
        end
    end
end
disp(B(2, 3, 4))

function s = sum_2d(A)
    s = 0;
    for i = 1:3
        for j = 1:3
            s = s + A(i, j);
        end
    end
end
