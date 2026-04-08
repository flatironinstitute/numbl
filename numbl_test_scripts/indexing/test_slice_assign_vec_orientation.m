% Test that subscripted slice assignment accepts row vs column vector
% orientation when the slice has a degenerate dimension.  MATLAB is
% lenient: A(rowVec, scalar) = rhs works with rhs as either row or
% column as long as the element count matches.  numbl used to reject
% the mismatched orientation with "Subscripted assignment dimension
% mismatch", breaking FLAM's permutation builders.

% --- Row RHS into a column slice ---
q = [3 1 2 4];
P = zeros(4, 2);
P(q, 1) = 1:4;
assert(isequal(P, [2 0; 3 0; 1 0; 4 0]), 'P(q,1)=row');

% --- Column RHS into a row slice ---
P2 = zeros(2, 4);
P2(1, q) = (1:4).';
assert(isequal(P2, [2 3 1 4; 0 0 0 0]), 'P2(1,q)=col');

% --- Matching shapes still work ---
P3 = zeros(4, 2);
P3(q, 1) = (1:4).';  % column into column slice
assert(isequal(P3, [2 0; 3 0; 1 0; 4 0]), 'col into col');

% --- Scalar RHS broadcasts ---
P4 = zeros(4, 2);
P4([1 3], 2) = 9;
assert(P4(1, 2) == 9 && P4(3, 2) == 9 && P4(2, 2) == 0, 'scalar broadcast');

% --- Vector slice (both dims non-degenerate) requires matching shape ---
threw = false;
try
    P5 = zeros(4, 4);
    P5([1 2], [3 4]) = 1:4;  % 1x4 vector into 2x2 slice — reject
catch
    threw = true;
end
assert(threw, '2x2 slice from 1x4 should error');

% --- Count-matches via reshape should still work ---
P6 = zeros(4, 4);
P6([1 2], [3 4]) = reshape(1:4, 2, 2);
assert(P6(1, 3) == 1 && P6(2, 3) == 2 && P6(1, 4) == 3 && P6(2, 4) == 4, ...
    'reshape into 2x2 slice');

disp('SUCCESS');
