% C-JIT parity gap #07: AssignIndexCol (dst(:, j) = src).
%
% The JS-JIT compiles `dst(:, j) = src` via setCol2r_h; the C-JIT
% historically bailed feasibility with
%   "unsupported stmt: AssignIndexCol"
% because the range/col family of writes had no C helpers.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 42\n-7\n99
%   numbl --opt 2 run <this>                         -> 42\n-7\n99  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 42\n-7\n99  (the gap is now closed)
%   matlab -batch parity07_col_slice_write           -> 42\n-7\n99
%
% Also asserts the caller's `V` stays unchanged through the write —
% AssignIndexCol uses the same unshare-at-entry path as parity03/06.

% 1) Basic column write: vals(:, j) = src where vals is 2x5.
vals = zeros(2, 5);
src = [42; 43];
V = set_col(vals, 3, src);
assert(isequal(vals, zeros(2, 5)), 'caller vals must be unchanged');
disp(V(1, 3))

% 2) Column index computed from a loop variable (j+1 style).
W = zeros(3, 4);
scratch = zeros(3, 1);
W2 = col_loop(W, scratch);
assert(isequal(W, zeros(3, 4)), 'caller W must be unchanged');
disp(W2(1, 3))   % expected: -7 (fill row 1 of col 3 with -7 on iter 2)

% 3) Length-mismatch (err-flag code 3.0): src is 3x1 into a 2-row dst.
threw = false;
try
    bad_col_write();
catch
    threw = true;
end
assert(threw, 'length mismatch should throw');
disp(99)

function V = set_col(V, j, src)
    V(:, j) = src;
end

function W = col_loop(W, src)
    for j = 1:3
        src(1) = -j * 3 - 1;
        src(2) = -j * 3 - 2;
        src(3) = -j * 3 - 3;
        W(:, j + 1) = src;
    end
end

function bad_col_write()
    dst = zeros(2, 4);
    src = [1; 2; 3];
    dst(:, 1) = src; %#ok<NASGU>
end
