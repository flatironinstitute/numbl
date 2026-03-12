% Test reductions on arrays with a leading singleton dimension [1,M,N].
% MATLAB's default for reductions is the first non-singleton dimension,
% which for [1,3,4] shaped arrays is dim 2 (not dim 1).

A = reshape(1:12, [1, 3, 4]);
% A(:,:,1) = [1 2 3]
% A(:,:,2) = [4 5 6]
% A(:,:,3) = [7 8 9]
% A(:,:,4) = [10 11 12]

%% sum along default dim (dim 2, first non-singleton)
B = sum(A);
assert(isequal(size(B), [1, 1, 4]));
assert(B(1,1,1) == 6);   % 1+2+3
assert(B(1,1,2) == 15);  % 4+5+6
assert(B(1,1,3) == 24);  % 7+8+9
assert(B(1,1,4) == 33);  % 10+11+12

%% prod along default dim (dim 2)
C = prod(A);
assert(isequal(size(C), [1, 1, 4]));
assert(C(1,1,1) == 6);    % 1*2*3
assert(C(1,1,2) == 120);  % 4*5*6

%% mean along default dim (dim 2)
D = mean(A);
assert(isequal(size(D), [1, 1, 4]));
assert(D(1,1,1) == 2);   % mean([1 2 3])
assert(D(1,1,2) == 5);   % mean([4 5 6])
assert(D(1,1,3) == 8);
assert(D(1,1,4) == 11);

%% max along default dim (dim 2)
E = max(A);
assert(isequal(size(E), [1, 1, 4]));
assert(E(1,1,1) == 3);
assert(E(1,1,2) == 6);
assert(E(1,1,3) == 9);
assert(E(1,1,4) == 12);

%% min along default dim (dim 2)
F = min(A);
assert(isequal(size(F), [1, 1, 4]));
assert(F(1,1,1) == 1);
assert(F(1,1,2) == 4);
assert(F(1,1,3) == 7);
assert(F(1,1,4) == 10);

%% [m,i] = min(A): min and index along default dim
[Fv, Fi] = min(A);
assert(isequal(size(Fv), [1, 1, 4]));
assert(isequal(size(Fi), [1, 1, 4]));
assert(Fv(1,1,1) == 1);
assert(Fi(1,1,1) == 1);  % index within dim 2
assert(Fv(1,1,2) == 4);
assert(Fi(1,1,2) == 1);

%% sort along default dim (dim 2)
G = reshape([3 1 2 6 4 5 9 7 8 12 10 11], [1, 3, 4]);
H = sort(G);
assert(isequal(size(H), [1, 3, 4]));
% page 1: sort [3,1,2] → [1,2,3]
assert(H(1,1,1) == 1);
assert(H(1,2,1) == 2);
assert(H(1,3,1) == 3);
% page 2: sort [6,4,5] → [4,5,6]
assert(H(1,1,2) == 4);
assert(H(1,2,2) == 5);
assert(H(1,3,2) == 6);

disp('SUCCESS');
