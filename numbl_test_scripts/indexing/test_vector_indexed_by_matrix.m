% Indexing a vector with a 2-D index matrix returns a result shaped like
% the index (MATLAB rule: result has idx's shape unless BOTH base and idx
% are vectors, in which case the result follows the base's orientation).

% row base, matrix index -> idx shape
vals = [10 20 30];
idx = [1 2; 3 1];
assert(isequal(vals(idx), [10 20; 30 10]));

% range base, matrix index
A = 1:5;
assert(isequal(A([1 2; 3 4]), [1 2; 3 4]));

% column base, matrix index -> still idx shape (base orientation ignored)
Ac = (1:5)';
assert(isequal(Ac([1 2; 3 4]), [1 2; 3 4]));

% both vectors -> result follows base orientation
colBase = (1:4)';
assert(isequal(colBase([2 3]), [2; 3]));   % column base, row idx -> column

rowBase = 1:4;
assert(isequal(rowBase([2; 3]), [2 3]));   % row base, column idx -> row

disp('SUCCESS')
