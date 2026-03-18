% Test element-wise division involving sparse and complex

%% complex sparse ./ complex dense
A = sparse([1+2i 0; 0 3+4i]);
B = [1+1i 2; 3 1-1i];
R = A ./ B;
expected = full(A) ./ B;
assert(max(max(abs(R - expected))) < 1e-10);

%% complex dense ./ complex sparse (non-zero divisor)
C = [2+1i 3; 4 1+2i];
D = sparse([1+1i 1; 1 2-1i]);
R2 = C ./ D;
expected2 = C ./ full(D);
assert(max(max(abs(R2 - expected2))) < 1e-10);

%% real scalar ./ sparse
S = sparse([2 0; 0 4]);
R3 = 8 ./ S;
expected3 = 8 ./ full(S);
assert(isequal(R3, expected3));

disp('SUCCESS')
