% Test spdiags builtin

% Test 1: main diagonal
S = spdiags([1;2;3], 0, 3, 3);
assert(S(1,1) == 1); assert(S(2,2) == 2); assert(S(3,3) == 3);
assert(S(1,2) == 0); assert(S(2,1) == 0);

% Test 2: superdiagonal d=1
% For d>0, MATLAB takes B(d+1:end), so B(2)=5 -> S(1,2), B(3)=6 -> S(2,3)
S = spdiags([4;5;6], 1, 3, 3);
assert(S(1,2) == 5); assert(S(2,3) == 6);
assert(S(1,1) == 0); assert(S(3,3) == 0);

% Test 3: subdiagonal d=-1
% d<0: B[j][k]->S[j-d][j] (0-indexed), so j=0: S[1][0]=B[0] -> S(2,1)=B(1,1)=1
%                                             j=1: S[2][1]=B[1] -> S(3,2)=B(2,1)=2
S = spdiags([1;2;3], -1, 3, 3);
assert(S(2,1) == 1); assert(S(3,2) == 2);
assert(S(1,1) == 0);

% Test 4: tridiagonal 4x4
B = [ones(4,1)*1, ones(4,1)*2, ones(4,1)*3];
S = spdiags(B, [-1 0 1], 4, 4);
assert(S(1,1) == 2); assert(S(2,2) == 2); assert(S(3,3) == 2); assert(S(4,4) == 2);
assert(S(2,1) == 1); assert(S(3,2) == 1); assert(S(4,3) == 1);
assert(S(1,2) == 3); assert(S(2,3) == 3); assert(S(3,4) == 3);
assert(S(1,3) == 0); assert(S(4,1) == 0);

% Test 5: non-square 3x5, diagonals 0 and 2
B = [ones(3,1), 2*ones(3,1)];
S = spdiags(B, [0 2], 3, 5);
assert(isequal(size(S), [3 5]));
assert(S(1,1) == 1); assert(S(2,2) == 1); assert(S(3,3) == 1);
assert(S(1,3) == 2); assert(S(2,4) == 2); assert(S(3,5) == 2);
assert(S(1,2) == 0); assert(S(1,4) == 0);

% Test 6: sptoeplitz-like usage with repmat
% Simulate the large-matrix path of sptoeplitz for a simple case:
% col = [5; 0; 0], row = [5; 0; 0] -> 3x3 diagonal matrix 5*eye(3)
col = [5; 0; 0];
row = [5; 0; 0];
[ic, ~, sc] = find(col);
row(1) = 0;
[ir, ~, sr] = find(row);
if ~isempty(ir)
    d = [ir - 1; 1 - ic];
    B = repmat([sr; sc].', 3, 1);
    T = spdiags(B, d, 3, 3);
else
    d = 1 - ic;
    B = repmat(sc.', 3, 1);
    T = spdiags(B, d, 3, 3);
end
assert(T(1,1) == 5); assert(T(2,2) == 5); assert(T(3,3) == 5);
assert(T(1,2) == 0);

disp('SUCCESS');
