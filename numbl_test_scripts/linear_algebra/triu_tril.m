% Test triu and tril builtins

% Test 1: triu of ones(4)
A = ones(4);
B = triu(A);
assert(B(1,1) == 1); assert(B(1,4) == 1); assert(B(4,4) == 1);
assert(B(2,1) == 0); assert(B(3,1) == 0); assert(B(4,1) == 0);
assert(B(3,2) == 0); assert(B(4,3) == 0);

% Test 2: triu(A,1) - strictly above main diagonal
C = triu(A,1);
assert(C(1,2) == 1); assert(C(1,4) == 1); assert(C(3,4) == 1);
assert(C(1,1) == 0); assert(C(2,2) == 0); assert(C(4,4) == 0);

% Test 3: triu(A,-1) - main diagonal and one subdiagonal
D = triu(A,-1);
assert(D(1,1) == 1); assert(D(2,1) == 1); assert(D(4,4) == 1);
assert(D(3,1) == 0); assert(D(4,1) == 0);

% Test 4: triu preserves values
M = [1 2 3; 4 5 6; 7 8 9];
U = triu(M);
assert(U(1,1) == 1); assert(U(1,2) == 2); assert(U(1,3) == 3);
assert(U(2,2) == 5); assert(U(2,3) == 6); assert(U(3,3) == 9);
assert(U(2,1) == 0); assert(U(3,1) == 0); assert(U(3,2) == 0);

% Test 5: tril of ones(4)
L = tril(A);
assert(L(1,1) == 1); assert(L(4,1) == 1); assert(L(4,4) == 1);
assert(L(1,2) == 0); assert(L(1,4) == 0); assert(L(2,3) == 0);

% Test 6: tril(A,-1) - strictly below main diagonal
L2 = tril(A,-1);
assert(L2(2,1) == 1); assert(L2(4,1) == 1); assert(L2(4,3) == 1);
assert(L2(1,1) == 0); assert(L2(3,3) == 0);

% Test 7: tril(A,1) - main diagonal and one superdiagonal
L3 = tril(A,1);
assert(L3(1,1) == 1); assert(L3(1,2) == 1); assert(L3(4,4) == 1);
assert(L3(1,3) == 0); assert(L3(2,4) == 0);

% Test 8: triu + tril = A + diag(diag(A))
N = [1 2 3; 4 5 6; 7 8 9];
assert(isequal(triu(N) + tril(N) - diag(diag(N)), N));

disp('SUCCESS')
