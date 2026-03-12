% Test matrix creation functions

% zeros
Z = zeros(3, 3);
assert(Z(1,1) == 0);
assert(Z(2,2) == 0);
assert(sum(sum(Z)) == 0);

% ones
O = ones(2, 4);
assert(size(O, 1) == 2);
assert(size(O, 2) == 4);
assert(sum(sum(O)) == 8);

% eye
I = eye(3);
assert(I(1,1) == 1);
assert(I(2,2) == 1);
assert(I(3,3) == 1);
assert(I(1,2) == 0);
assert(I(2,3) == 0);

% size and numel
A = zeros(4, 5);
assert(size(A, 1) == 4);
assert(size(A, 2) == 5);
assert(numel(A) == 20);

% colon operator as vector
v = 1:5;
assert(length(v) == 5);
assert(v(1) == 1);
assert(v(5) == 5);

% colon with step
v2 = 0:2:10;
assert(length(v2) == 6);
assert(v2(3) == 4);

disp('SUCCESS')
