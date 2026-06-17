% setdiff(A, B, 'rows') — row-wise set difference: the unique rows of A
% not present in B, sorted ascending. Used by DistMesh's distmesh2d
% (`setdiff(p, pfix, 'rows')`) and simpplot.

%% Basic row difference keeps the matrix shape (N x cols)
A = [-1 -1; 0 0; 0.5 0.5; 1 1; 0.3 0.7];
B = [-1 -1; -1 1; 1 -1; 1 1];
C = setdiff(A, B, 'rows');
assert(isequal(C, [0 0; 0.3 0.7; 0.5 0.5]), 'rows not in B, sorted');
assert(size(C, 2) == 2, 'result keeps 2 columns');

%% [C, IA] gives C = A(IA, :)
[C2, ia] = setdiff(A, B, 'rows');
assert(isequal(C2, A(ia, :)), 'C = A(IA,:) relationship');

%% duplicate rows in A collapse to one
D = [1 2; 1 2; 3 4];
E = [9 9];
assert(isequal(setdiff(D, E, 'rows'), [1 2; 3 4]), 'duplicate rows collapse');

%% all rows shared -> empty result with the right column count
F = setdiff([1 1; 2 2], [1 1; 2 2; 3 3], 'rows');
assert(isempty(F), 'all rows shared -> empty');

%% three-column rows
G = setdiff([1 2 3; 4 5 6; 7 8 9], [4 5 6], 'rows');
assert(isequal(G, [1 2 3; 7 8 9]), '3-column rows');

disp('SUCCESS')
