% Test that adding a real number to a complex number works correctly
% (not treated as string concatenation)

x = [1+2i, 3+4i];
s = sum(x);
assert(~isreal(s));

y = 1 + s;
assert(strcmp(class(y), 'double'));
assert(~isreal(y));
assert(abs(real(y) - 5) < 1e-10);
assert(abs(imag(y) - 6) < 1e-10);

% Also test the reverse order
z = s + 1;
assert(strcmp(class(z), 'double'));
assert(abs(real(z) - 5) < 1e-10);

fprintf('SUCCESS\n');
