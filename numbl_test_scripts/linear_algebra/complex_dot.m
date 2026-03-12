% Test dot() with complex input
% MATLAB dot product is conjugate dot: dot(a,b) = sum(conj(a) .* b)

% Vector case: dot([1+2i, 3+4i], [5+6i, 7+8i])
% conj([1+2i,3+4i]) = [1-2i, 3-4i]
% (1-2i)*(5+6i) = 5+6i-10i-12i^2 = 5-4i+12 = 17-4i
% (3-4i)*(7+8i) = 21+24i-28i-32i^2 = 21-4i+32 = 53-4i
% sum = 70-8i
a = [1+2i, 3+4i];
b = [5+6i, 7+8i];
d = dot(a, b);
assert(abs(real(d) - 70) < 1e-10);
assert(abs(imag(d) - (-8)) < 1e-10);

% Pure imaginary vectors
a2 = [1i, 2i];
b2 = [3i, 4i];
% conj([1i,2i]) = [-1i,-2i]
% (-1i)*(3i) = -3i^2 = 3
% (-2i)*(4i) = -8i^2 = 8
% sum = 11
d2 = dot(a2, b2);
assert(abs(real(d2) - 11) < 1e-10);
assert(abs(imag(d2)) < 1e-10);

disp('SUCCESS')
