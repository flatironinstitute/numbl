% Cell-element type tracking must invalidate when a non-literal-index
% cell write happens. The non-literal write COULD overwrite any slot,
% so all tracked element types in the cell go unknown.
%
% Pre-fix bug: literal write `c{1} = 5` tracked c{1} as number;
% then `c{k} = tensor` (non-literal) didn't update tracking; then
% `x = c{1}; y = x + 1` generated scalar JS `+`, which on a runtime
% tensor produced string concatenation garbage.

c = cell(3, 1);
n = 2;
big_tensor = ones(4, 1) * 3;
s = 0;
for k = 1:n
    c{1} = 5;
    c{k} = big_tensor;     % non-literal write — at k==1 c{1} actually changes
    x = c{1};
    y = x + 1;             % must NOT use scalar JS `+` (would mishandle tensor)
    s = s + sum(y);
end

% k=1: c{1} becomes big_tensor (4 elements of 3), y = 4*1 vector of 4s, sum=16
% k=2: c{1} stays as the scalar 5, y = 6, sum = 6
expected = 16 + 6;
assert(abs(s - expected) < 1e-9, ...
    sprintf('s should be %d (got %g) — non-literal cell write must invalidate element tracking', expected, s));

disp('SUCCESS')
