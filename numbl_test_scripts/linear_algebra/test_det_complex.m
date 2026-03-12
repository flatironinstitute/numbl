% Test det with complex matrices
% Bug: det returns wrong value for complex matrices

% Upper triangular complex matrix: det = product of diagonal
F = [1+1i, 2; 0, 1-1i];
d = det(F);
% (1+i)(1-i) = 1 - i^2 = 1 + 1 = 2
assert(abs(real(d) - 2) < 1e-10);
assert(abs(imag(d)) < 1e-10);

% Diagonal complex matrix
G = [2+1i, 0; 0, 3-1i];
d2 = det(G);
% (2+i)(3-i) = 6 - 2i + 3i - i^2 = 6 + i + 1 = 7 + i
assert(abs(real(d2) - 7) < 1e-10);
assert(abs(imag(d2) - 1) < 1e-10);

% General complex matrix
H = [1+1i, 2+1i; 3, 4-2i];
d3 = det(H);
% (1+i)(4-2i) - (2+i)(3) = (4-2i+4i-2i^2) - (6+3i) = (4+2i+2) - (6+3i) = 6+2i-6-3i = -i
assert(abs(real(d3)) < 1e-10);
assert(abs(imag(d3) - (-1)) < 1e-10);

disp('SUCCESS');
