% C-JIT parity gap #05: multi-index (2D/3D) tensor AssignIndex write.
%
% The JS-JIT compiles `A(i,j) = v` / `B(i,j,k) = v` via set2r_h /
% set3r_h (soft-bails on OOB so the interpreter can grow the tensor);
% the C-JIT historically bailed feasibility with
%   "multi-index AssignIndex not supported"
% because multi-index writes need the tensor's shape dims in the ABI
% and numbl_jit_runtime only had numbl_set1r_h.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 42\n234
%   numbl --opt 2 run <this>                         -> 42\n234  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 42\n234  (the gap is now closed)
%   matlab -batch parity05_multi_index_write         -> 42\n234
%
% Also asserts that the caller's A stays unchanged after a 2D write
% through a pure-input tensor param — the C-JIT routes this through the
% same unshare-at-entry path that parity03 established for 1D writes.

A = [1 2 3; 4 5 6; 7 8 9];
B = set_2d(A, 2, 3, 42);
assert(isequal(A, [1 2 3; 4 5 6; 7 8 9]), 'caller A must be unchanged');
disp(B(2, 3))

% The 3D fill runs with a pre-allocated tensor passed in — that way the
% jitted function only exercises the multi-index AssignIndex path we're
% testing here (zeros() with a 3D shape isn't in the C-JIT feasibility
% whitelist, so we allocate at script level).
C0 = zeros(2, 3, 4);
C = fill_3d(C0);
disp(C(2, 3, 4))

function out = set_2d(A, i, j, v)
    A(i, j) = v;
    out = A;
end

function B = fill_3d(B)
    for i = 1:2
        for j = 1:3
            for k = 1:4
                B(i, j, k) = i * 100 + j * 10 + k;
            end
        end
    end
end
