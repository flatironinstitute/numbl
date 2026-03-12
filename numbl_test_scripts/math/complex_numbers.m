% Test complex number operations

% Basic complex number
z = 3 + 4i;
assert(real(z) == 3);
assert(imag(z) == 4);

% abs (magnitude)
assert(abs(z) == 5);

% Arithmetic with complex
z2 = 1 - 2i;
zs = z + z2;
assert(real(zs) == 4);
assert(imag(zs) == 2);

zp = z * z2;
% (3+4i)(1-2i) = 3-6i+4i-8i^2 = 3-2i+8 = 11-2i
assert(real(zp) == 11);
assert(imag(zp) == -2);

% conj
zc = conj(z);
assert(real(zc) == 3);
assert(imag(zc) == -4);

% sqrt(-1)
zi = sqrt(-1);
assert(abs(real(zi)) < 1e-5);
assert(abs(imag(zi) - 1) < 1e-5);

disp('SUCCESS')
