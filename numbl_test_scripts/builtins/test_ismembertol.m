% ismembertol: default tolerance, scaling, LocB choice, ByRows, options

A = [1 2 3+1e-7 5]; B = [3 1 2];
[l, i] = ismembertol(A, B);
assert(isequal(l, logical([1 1 0 0])) && isequal(i, [2 3 0 0]));
assert(islogical(l));
[l, i] = ismembertol(A, B, 1e-6);
assert(isequal(l, logical([1 1 1 0])) && isequal(i, [2 3 1 0]));

% tolerance is scaled by max(abs([A(:); B(:)]))
assert(~ismembertol(3+4e-6, B, 1e-6));
assert(ismembertol(3+4e-6, [3 1 2 10], 1e-6));
assert(ismembertol(3+4e-6, B, 1e-6, 'DataScale', 5));

% several matches: LocB is the smallest matching element of B
[~, i] = ismembertol(1, [5 1+1e-8 1 1-1e-8], 1e-6);
assert(i == 4);
[~, i] = ismembertol(1, [5 1-1e-8 1 1+1e-8], 1e-6);
assert(i == 2);

% ByRows (lexicographically smallest matching row)
A = [1 2; 3 4; 5 6]; B = [3 4+1e-7; 1 2; 3 4];
[l, i] = ismembertol(A, B, 1e-6, 'ByRows', true);
assert(isequal(l, logical([1; 1; 0])) && isequal(i, [2; 3; 0]));

% ByRows scales each column separately
assert(~ismembertol([1e-3 100], [1.5e-3 100], 1e-4, 'ByRows', true));
[l, i] = ismembertol([1 2; 3 4], [3 4], 0.1, 'ByRows', true, 'DataScale', [1 1]);
assert(isequal(l, logical([0; 1])) && isequal(i, [0; 1]));

% shape follows A
[l, i] = ismembertol([1 2; 3 9], [3 1 2], 1e-6);
assert(isequal(l, logical([1 1; 1 0])) && isequal(i, [2 3; 1 0]));
assert(isempty(ismembertol([], [1 2])));

% NaN is never a member
[l, i] = ismembertol([NaN 1], [NaN 1]);
assert(isequal(l, logical([0 1])) && isequal(i, [0 2]));

% OutputAllIndices
[l, i] = ismembertol([1 2 7], [1 1 2 3], 1e-6, 'OutputAllIndices', true);
assert(iscell(i) && isequal(i{1}, [1; 2]) && isequal(i{2}, 3) && isequal(i{3}, 0));

disp('SUCCESS')
