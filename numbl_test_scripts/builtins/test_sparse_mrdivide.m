% Test sparse matrix right division (/)

%% sparse / dense matrix
S = sparse([1 2; 3 4]);
D = [1 0; 0 1];
R = S / D;
assert(isequal(R, full(S) / D));

%% sparse / scalar (should return sparse)
S2 = sparse([2 0; 0 4]);
R2 = S2 / 2;
assert(isequal(full(R2), [1 0; 0 2]));

%% complex sparse / dense matrix
CS = sparse([1+1i 0; 0 2+2i]);
R3 = CS / eye(2);
assert(isequal(R3, full(CS)));

disp('SUCCESS')
