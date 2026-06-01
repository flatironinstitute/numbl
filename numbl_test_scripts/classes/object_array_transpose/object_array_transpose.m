% Test: transpose / ctranspose and 2-D shape for default object arrays (a
% class that overloads none of horzcat/vertcat/transpose/ctranspose). All
% expected values were verified against MATLAB R2025b.

a = OBox(1); b = OBox(2); c = OBox(3); d = OBox(4);

% Row vector and its transpose (-> column).
row = [a b c];
assert(isequal(size(row), [1 3]), 'row size should be [1 3]');
rt = row.';
assert(isequal(size(rt), [3 1]), 'row.'' size should be [3 1]');
assert(rt(1).v == 1 && rt(2).v == 2 && rt(3).v == 3, 'row.'' values');

% Column vector (vertcat) and its transpose (-> row).
col = [a; b; c];
assert(isequal(size(col), [3 1]), 'col size should be [3 1]');
assert(col(1).v == 1 && col(3).v == 3, 'col values');
assert(isequal(size(col.'), [1 3]), 'col.'' size should be [1 3]');

% 2-D literal [a b; c d] -> 2x2, column-major elements a,c,b,d.
M = [a b; c d];
assert(isequal(size(M), [2 2]), 'M size should be [2 2]');
assert(M(2,1).v == 3, 'M(2,1) should be c=3');
assert(M(1,2).v == 2, 'M(1,2) should be b=2');
assert(M(1).v == 1 && M(2).v == 3 && M(3).v == 2 && M(4).v == 4, ...
    'M linear (column-major) should be a,c,b,d');

% Transpose permutes: Mt(i,j) = M(j,i).
Mt = M.';
assert(isequal(size(Mt), [2 2]), 'M.'' size should be [2 2]');
assert(Mt(1,2).v == 3, 'Mt(1,2) should equal M(2,1)=c=3');
assert(Mt(2,1).v == 2, 'Mt(2,1) should equal M(1,2)=b=2');

% ctranspose behaves like transpose for objects (no element conjugation).
Mc = M';
assert(isequal(size(Mc), [2 2]) && Mc(1,2).v == 3, 'ctranspose matches transpose');

% Scalar object transpose is an identity.
s = a.';
assert(isequal(size(s), [1 1]) && s.v == 1, 'scalar transpose is identity');

% Subset indexing preserves source orientation.
assert(isequal(size(row([1 3])), [1 2]), 'row subset stays a row');
assert(isequal(size(col([1 3])), [2 1]), 'col subset stays a column');

% length is max(size); numel is the element count.
assert(length(M) == 2, 'length(M) should be 2');
assert(numel(M) == 4, 'numel(M) should be 4');

disp('SUCCESS')
