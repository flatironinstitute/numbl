% Test that dot product of complex vectors returns correct type
% and that arithmetic on the result works correctly

a = [1+1i, 2+2i];
b = [1-1i, 1+1i];

% dot(a,b) should be complex in general
d = dot(a, b);
y = d + 1;
assert(abs(real(y) - real(d) - 1) < 1e-10);
assert(abs(imag(y) - imag(d)) < 1e-10);

fprintf('SUCCESS\n');
