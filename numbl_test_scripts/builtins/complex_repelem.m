% Test repelem() with complex tensor input
v = [1+2i, 3+4i];
r = repelem(v, 2);
assert(size(r, 2) == 4);
assert(abs(real(r(1)) - 1) < 1e-10);
assert(abs(imag(r(1)) - 2) < 1e-10);
assert(abs(real(r(2)) - 1) < 1e-10);
assert(abs(imag(r(2)) - 2) < 1e-10);
assert(abs(real(r(3)) - 3) < 1e-10);
assert(abs(imag(r(3)) - 4) < 1e-10);
assert(abs(real(r(4)) - 3) < 1e-10);
assert(abs(imag(r(4)) - 4) < 1e-10);

% Matrix case: repelem with complex matrix
M = [1+2i, 3+4i; 5+6i, 7+8i];
R = repelem(M, 2, 2);
assert(size(R, 1) == 4);
assert(size(R, 2) == 4);
assert(abs(real(R(1,1)) - 1) < 1e-10);
assert(abs(imag(R(1,1)) - 2) < 1e-10);
assert(abs(real(R(3,3)) - 7) < 1e-10);
assert(abs(imag(R(3,3)) - 8) < 1e-10);

disp('SUCCESS')
