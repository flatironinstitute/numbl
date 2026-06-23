% char() of a cell array of strings builds a space-padded char matrix;
% lower/upper must preserve that multi-row shape.

C = {'TraceSolution'; 'Tolerance'; 'SolverMethod'; 'InitialGuess'};
m = char(C);
assert(isequal(size(m), [4 13]), 'char(cell) size');
assert(strcmp(strtrim(m(2, :)), 'Tolerance'), 'row 2 content');
assert(m(2, 10) == ' ', 'row 2 padded with spaces');

% lower / upper preserve the 4x13 shape
lo = lower(m);
assert(isequal(size(lo), [4 13]), 'lower keeps shape');
assert(strcmp(strtrim(lo(1, :)), 'tracesolution'), 'lower row 1');
up = upper(m);
assert(isequal(size(up), [4 13]), 'upper keeps shape');
assert(strcmp(strtrim(up(3, :)), 'SOLVERMETHOD'), 'upper row 3');

% Single-element cell collapses to a row vector
one = char({'abc'});
assert(isequal(size(one), [1 3]), 'char({single}) size');
assert(strcmp(one, 'abc'), 'char({single}) content');

% Rows of differing lengths all pad to the widest
m2 = char({'a'; 'bb'; 'ccc'});
assert(isequal(size(m2), [3 3]), 'ragged size');
assert(strcmp(m2(1, :), 'a  '), 'ragged padding');

disp('SUCCESS');
