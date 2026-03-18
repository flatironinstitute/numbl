% Test spconvert builtin

% Test 1: basic triplet conversion
S = [1 2 3; 2 3 4; 5 5 0];
A = spconvert(S);
assert(issparse(A));
assert(full(A(1,2)) == 3);
assert(full(A(2,3)) == 4);
[m, n] = size(A);
assert(m == 5);
assert(n == 5);

% Test 2: simple 3x3
S2 = [1 1 10; 2 2 20; 3 3 30];
A2 = spconvert(S2);
assert(full(A2(1,1)) == 10);
assert(full(A2(2,2)) == 20);
assert(full(A2(3,3)) == 30);
assert(nnz(A2) == 3);

% Test 3: equivalent to sparse(i,j,v,m,n)
i = [1; 3; 2];
j = [2; 1; 3];
v = [5; 6; 7];
A3 = spconvert([i j v; 4 4 0]);
A3_ref = sparse(i, j, v, 4, 4);
assert(nnz(A3 - A3_ref) == 0);

disp('SUCCESS');
