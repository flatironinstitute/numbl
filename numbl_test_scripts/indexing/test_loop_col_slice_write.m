% Multi-dim column slice write `dst(:, j) = src` inside a loop —
% exercises the loop-JIT path for AssignIndexCol. Mirrors chunkie's
% adapgausskerneval column stack write `vals(:, jj+1) = v2;`.
%
% Cases below probe:
%   - basic column write from a column-vector source
%   - interleaved scalar writes and column writes to the same dst
%   - column write with a non-trivial index expression
%   - column write where src is a sliced alias of another tensor
%   - error when RHS length doesn't match dst.rows
%   - auto-growth when column index exceeds current dst.cols (MATLAB-compatible)

% 1) Basic column write: dst(:, j) = src
vals = zeros(2, 5);
src = zeros(2, 1);
for j = 1:5
    src(1) = j * 10;
    src(2) = j * 10 + 1;
    vals(:, j) = src;
end
assert(vals(1, 1) == 10 && vals(2, 1) == 11, '1: col 1 mismatch');
assert(vals(1, 5) == 50 && vals(2, 5) == 51, '1: col 5 mismatch');

% 2) Interleaved scalar + column writes to the same tensor
stack = zeros(2, 10);
v = zeros(2, 1);
for i = 1:5
    v(1) = i;
    v(2) = -i;
    stack(:, i) = v;
    stack(1, i + 5) = i * 100;
end
assert(stack(1, 3) == 3 && stack(2, 3) == -3, '2: col write at i=3');
assert(stack(1, 8) == 300, '2: scalar write into col 8 row 1');
assert(stack(2, 8) == 0, '2: col 8 row 2 untouched');

% 3) Column index computed from loop variable (j+1 style)
dst = zeros(3, 8);
s = zeros(3, 1);
for i = 1:7
    s(1) = i;
    s(2) = i * 2;
    s(3) = i * 3;
    dst(:, i + 1) = s;
end
assert(dst(1, 1) == 0 && dst(2, 1) == 0 && dst(3, 1) == 0, '3: col 1 zero');
assert(dst(1, 2) == 1 && dst(2, 2) == 2 && dst(3, 2) == 3, '3: col 2');
assert(dst(1, 8) == 7 && dst(3, 8) == 21, '3: col 8');

% 4) Source length mismatch should throw
bad_src = zeros(3, 1);
bad_dst = zeros(2, 4);
ok = false;
try
    for j = 1:1
        bad_dst(:, j) = bad_src;
    end
catch
    ok = true;
end
assert(ok, '4: expected size mismatch error');

% 5) Column index past current dst.cols auto-grows the tensor (MATLAB
%    semantics — there's no "out of bounds" on writes, only on reads).
grow_src = ones(2, 1) * 7;
grow_dst = zeros(2, 3);
for j = 1:1
    grow_dst(:, 5) = grow_src;
end
assert(size(grow_dst, 1) == 2 && size(grow_dst, 2) == 5, ...
    '5: dst should have grown to 2x5');
assert(grow_dst(1, 5) == 7 && grow_dst(2, 5) == 7, '5: col 5 mismatch');
assert(grow_dst(1, 4) == 0 && grow_dst(2, 4) == 0, '5: col 4 should stay zero');

disp('SUCCESS');
