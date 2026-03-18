% Test sparse + complex scalar and complex scalar + sparse

%% sparse + complex scalar
S = sparse([1 0; 0 2]);
R = S + (1+2i);
expected = [1 0; 0 2] + (1+2i);
assert(isequal(R, expected));

%% complex scalar + sparse
R2 = (1+2i) + S;
assert(isequal(R2, expected));

%% sparse - complex scalar
R3 = S - (1+2i);
expected3 = [1 0; 0 2] - (1+2i);
assert(isequal(R3, expected3));

%% complex scalar - sparse
R4 = (3+4i) - S;
expected4 = (3+4i) - [1 0; 0 2];
assert(isequal(R4, expected4));

disp('SUCCESS')
