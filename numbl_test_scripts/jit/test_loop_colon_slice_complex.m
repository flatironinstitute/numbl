% 2D colon-slice read on a complex-valued tensor inside a JIT loop:
%   z = A(:, j)  or  z = A(j, :)  with A complex.
% This is the chunkie adapgausskerneval inner-loop pattern
%   dd = max(abs(v2 + v3 - vals(:, jj)));
% where `vals` is a complex 1xN scratch buffer. Without complex
% support in `__extractSlice2d`, the lowering bails at the `(:, j)`
% colon and the surrounding loop falls back to the interpreter.
%
% Real-tensor cases are covered by other tests; here we pin the
% complex shapes so a regression on extractSlice2d won't slip
% through.

% Build a complex 4x6 tensor whose entries are deterministic.
m = 4; nc = 6;
A = zeros(m, nc) + 1i * zeros(m, nc);
for r = 1:m
    for c = 1:nc
        A(r, c) = (r + c) + 1i * (r - c);
    end
end

% --- 1) Column slice A(:, j) — fix column, vary row.
acc1 = 0 + 0i;
for j = 1:nc
    %!numbl:assert_jit
    col = A(:, j);
    acc1 = acc1 + sum(col);
end
expected1 = sum(A(:));
assert(abs(acc1 - expected1) < 1e-12, '1: column slice complex sum');

% --- 2) Row slice A(j, :) — fix row, vary column.
acc2 = 0 + 0i;
for r = 1:m
    %!numbl:assert_jit
    row = A(r, :);
    acc2 = acc2 + sum(row);
end
assert(abs(acc2 - expected1) < 1e-12, '2: row slice complex sum');

% Complex slice WRITES (e.g. `B(:, k) = ...` on a complex B) go
% through a different lowering path (lowerAssignLValue / page-write
% helpers) and stay bailed for now — covered by a separate follow-up.

disp('SUCCESS')
