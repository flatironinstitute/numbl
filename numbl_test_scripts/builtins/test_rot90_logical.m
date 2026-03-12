% Test that rot90 preserves logical type
N = 4;
x = logical(triu(ones(N)));
y = rot90(x);
assert(islogical(y));
assert(size(y, 1) == N);
assert(size(y, 2) == N);

% rot90 with k=2
y2 = rot90(x, 2);
assert(islogical(y2));

% rot90 with k=0
y0 = rot90(x, 0);
assert(islogical(y0));

% Logical indexing assignment using rot90 result
Dx = ones(N);
DxRot = rot90(Dx, 2);
idxTo = rot90(~triu(ones(N)));
assert(islogical(idxTo));
Dx(idxTo) = -DxRot(idxTo);
assert(Dx(2, 4) == -1);  % lower-right triangle should be -1
assert(Dx(4, 1) == 1);   % lower-left triangle should remain 1

fprintf('SUCCESS\n');
