% Regression test: trace() must preserve the imaginary part of a complex
% matrix. (It previously summed only the real diagonal, which broke the
% non-Hermitian eigenvector reconstruction in arXiv:1911.04906.)
tol = 1e-12;

% Complex matrix: trace = sum of complex diagonal entries.
A = [1+2i, 3+4i; 5+6i, 7+8i];
t = trace(A);
assert(abs(real(t) - 8) < tol, 'real(trace) should be 8');
assert(abs(imag(t) - 10) < tol, 'imag(trace) should be 10');

% trace(L*R) on complex matrices must match the explicit diagonal sum.
L = [0.518875, -0.481209-0.116747i; 0.481209+0.116747i, -0.454015-0.185081i];
R = [0.499375-0.025i, 0.5; -0.5, -0.499375+0.025i];
P = L*R;
t2 = trace(P);
t2d = P(1,1) + P(2,2);
assert(abs(t2 - t2d) < 1e-10, 'trace(L*R) must equal P(1,1)+P(2,2)');
assert(abs(imag(t2)) > 0.1, 'trace(L*R) must keep a nonzero imaginary part');

% Purely real result from a complex matrix collapses to real.
B = [1+1i, 0; 0, 2-1i];
tb = trace(B);
assert(abs(tb - 3) < tol, 'trace should be 3');
assert(abs(imag(tb)) < tol, 'imag part should vanish');

% Complex scalar: trace of a 1x1 is itself.
assert(abs(trace(3+4i) - (3+4i)) < tol, 'trace of complex scalar');

disp('SUCCESS');
