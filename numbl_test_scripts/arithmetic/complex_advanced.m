% Test: complex() builtin, sign() on complex, .^ on complex tensors

% complex() function - create complex numbers
z1 = complex(3, 4);
assert(real(z1) == 3);
assert(imag(z1) == 4);

z2 = complex(5);
assert(real(z2) == 5);
assert(imag(z2) == 0);

z3 = complex([1 2 3], [4 5 6]);
assert(z3(1) == 1+4i);
assert(z3(2) == 2+5i);
assert(z3(3) == 3+6i);

% sign() on complex numbers - should return z/abs(z)
s1 = sign(1+1i);
expected = (1+1i)/sqrt(2);
assert(abs(s1 - expected) < 1e-10);

s2 = sign(3+4i);
assert(abs(s2 - (3+4i)/5) < 1e-10);

s3 = sign(0+0i);
assert(s3 == 0);

% Element-wise power on complex tensors
A = [1+2i, 3+4i; 5+6i, 7+8i];
P = A .^ 2;
% (1+2i)^2 = 1+4i-4 = -3+4i
assert(abs(P(1,1) - (-3+4i)) < 1e-10);
% (3+4i)^2 = 9+24i-16 = -7+24i
assert(abs(P(1,2) - (-7+24i)) < 1e-10);

% .^ with complex exponent
Q = [4, 9, 16] .^ 0.5;
assert(abs(Q(1) - 2) < 1e-10);
assert(abs(Q(2) - 3) < 1e-10);
assert(abs(Q(3) - 4) < 1e-10);

disp('SUCCESS');
