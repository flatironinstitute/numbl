% setdiff / intersect / union with the 'rows' option and setOrder
% ('sorted' default | 'stable'), plus the index outputs (ia / ib).
% Matches the MATLAB docs. setdiff(...,'rows') is used by DistMesh.

%% ── setdiff 'rows' ────────────────────────────────────────────────────
A = [-1 -1; 0 0; 0.5 0.5; 1 1; 0.3 0.7];
B = [-1 -1; -1 1; 1 -1; 1 1];
C = setdiff(A, B, 'rows');
assert(isequal(C, [0 0; 0.3 0.7; 0.5 0.5]), 'setdiff rows sorted');
assert(size(C, 2) == 2, 'setdiff rows keeps columns');

[C2, ia] = setdiff(A, B, 'rows');
assert(isequal(C2, A(ia, :)), 'setdiff rows: C = A(ia,:)');

% all rows shared -> empty result
assert(isempty(setdiff([1 1; 2 2], [1 1; 2 2; 3 3], 'rows')), 'setdiff rows empty');

% three columns
assert(isequal(setdiff([1 2 3; 4 5 6; 7 8 9], [4 5 6], 'rows'), [1 2 3; 7 8 9]), '3-col rows');

%% ── setOrder 'stable' (element-wise) ──────────────────────────────────
assert(isequal(setdiff([5 1 3 1], 3, 'stable'), [5 1]), 'setdiff stable');
assert(isequal(intersect([7 1 5], [5 7 9], 'stable'), [7 5]), 'intersect stable');
assert(isequal(union([5 7 1], [3 1 1], 'stable'), [5 7 1 3]), 'union stable');

%% ── intersect indices (C = A(ia) = B(ib)) ─────────────────────────────
A2 = [7 1 5];
B2 = [5 7 9];
[ci, ia2, ib2] = intersect(A2, B2);
assert(isequal(ci, [5 7]), 'intersect sorted');
assert(isequal(A2(ia2), ci), 'intersect C = A(ia)');
assert(isequal(B2(ib2), ci), 'intersect C = B(ib)');

%% ── union indices (C = sort([A(ia) B(ib)])) ───────────────────────────
A3 = [5 7 1];
B3 = [3 1 1];
[cu, ia3, ib3] = union(A3, B3);
assert(isequal(cu, [1 3 5 7]), 'union sorted');
assert(isequal(sort([A3(ia3) B3(ib3)]), cu), 'union C = sort([A(ia) B(ib)])');

%% ── intersect / union 'rows' ──────────────────────────────────────────
[cr, iar, ibr] = intersect([1 2; 3 4; 5 6], [3 4; 7 8; 1 2], 'rows');
assert(isequal(cr, [1 2; 3 4]), 'intersect rows');
assert(isequal([1 2; 3 4; 5 6](iar, :), cr), 'intersect rows C = A(ia,:)');

assert(isequal(union([1 2; 3 4], [3 4; 5 6], 'rows'), [1 2; 3 4; 5 6]), 'union rows');

disp('SUCCESS');
