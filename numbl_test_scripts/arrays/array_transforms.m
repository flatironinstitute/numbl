% Test array transformation functions

% fliplr
v = [1, 2, 3, 4, 5];
vf = fliplr(v);
assert(vf(1) == 5);
assert(vf(3) == 3);
assert(vf(5) == 1);

% flipud
A = [1, 2; 3, 4; 5, 6];
Af = flipud(A);
assert(Af(1,1) == 5);
assert(Af(2,1) == 3);
assert(Af(3,1) == 1);

% repmat
B = repmat([1, 2], 1, 3);
assert(size(B, 2) == 6);
assert(B(1) == 1);
assert(B(2) == 2);
assert(B(5) == 1);
assert(B(6) == 2);

% transpose
C = [1, 2, 3; 4, 5, 6];
Ct = C';
assert(size(Ct, 1) == 3);
assert(size(Ct, 2) == 2);
assert(Ct(1,1) == 1);
assert(Ct(2,1) == 2);
assert(Ct(1,2) == 4);

disp('SUCCESS')
