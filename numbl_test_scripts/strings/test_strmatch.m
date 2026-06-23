% strmatch (legacy): find rows that begin with (or, with 'exact', equal) a
% pattern. Returns a column vector of matching row indices.

% Cell array of strings
assert(isequal(strmatch('trust', {'line', 'trust'}), 2), 'cell match');

% Char matrix, prefix match
names = char({'tracesolution'; 'tolerance'; 'solvermethod'; 'initialguess'});
assert(isequal(strmatch('tol', names), 2), 'prefix tol');
assert(isequal(strmatch('t', names), [1; 2]), 'prefix t (column vector)');

% 'exact' flag
assert(isempty(strmatch('tol', names, 'exact')), 'exact tol no match');
assert(isequal(strmatch('tolerance', names, 'exact'), 2), 'exact full');

% No match returns 0x1 empty
r = strmatch('zzz', names);
assert(isempty(r), 'no match empty');
assert(isequal(size(r), [0 1]), 'no match is 0x1');

% Pattern longer than any row -> empty
assert(isempty(strmatch('toleranceXYZ', names)), 'too-long pattern');

disp('SUCCESS');
