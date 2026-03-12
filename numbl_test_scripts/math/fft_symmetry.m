% Test that fft of real input produces conjugate-symmetric output,
% and ifft of conjugate-symmetric input produces pure real output.
% These are structural properties — we check isreal() and exact symmetry,
% not just numerical tolerance.

tol = 1e-10;

% --- fft of real input should have exact conjugate symmetry ---

% Test 1: large real vector (power-of-2) — fft should be conjugate-symmetric
x1 = randn(256, 1);
X1 = fft(x1);
N1 = length(X1);
% DC component should be exactly real
assert(imag(X1(1)) == 0);
% Nyquist (N/2+1) should be exactly real for even N
assert(imag(X1(N1/2 + 1)) == 0);
% Conjugate symmetry: X(k) == conj(X(N-k+2)) exactly
for k = 2:N1/2
    partner = N1 - k + 2;
    assert(real(X1(k)) == real(X1(partner)));
    assert(imag(X1(k)) == -imag(X1(partner)));
end

% Test 2: large real vector (non-power-of-2, uses DFT path)
x2 = randn(100, 1);
X2 = fft(x2);
N2 = length(X2);
assert(imag(X2(1)) == 0);
for k = 2:N2
    partner = mod(N2 - k + 1, N2) + 1;
    assert(real(X2(k)) == real(X2(partner)));
    assert(imag(X2(k)) == -imag(X2(partner)));
end

% Test 3: real row vector should also have conjugate symmetry
x3 = randn(1, 128);
X3 = fft(x3);
N3 = length(X3);
assert(imag(X3(1)) == 0);
for k = 2:N3/2
    partner = N3 - k + 2;
    assert(real(X3(k)) == real(X3(partner)));
    assert(imag(X3(k)) == -imag(X3(partner)));
end

% Test 4: real vector with zero-padding should have conjugate symmetry
x4 = randn(50, 1);
X4 = fft(x4, 128);
N4 = length(X4);
assert(imag(X4(1)) == 0);
for k = 2:N4/2
    partner = N4 - k + 2;
    assert(real(X4(k)) == real(X4(partner)));
    assert(imag(X4(k)) == -imag(X4(partner)));
end

% --- ifft of conjugate-symmetric input should be pure real ---

% Test 5: ifft(fft(real)) should be isreal for large vectors
x5 = randn(512, 1);
y5 = ifft(fft(x5));
assert(isreal(y5));
assert(max(abs(y5 - x5)) < tol);

% Test 6: ifft(fft(real)) for non-power-of-2
x6 = randn(200, 1);
y6 = ifft(fft(x6));
assert(isreal(y6));
assert(max(abs(y6 - x6)) < tol);

% Test 7: ifft of manually constructed conjugate-symmetric spectrum
N7 = 8;
X7 = zeros(N7, 1);
X7(1) = 10;
X7(2) = 3 - 2i;
X7(3) = -1 + 1i;
X7(4) = 0.5 - 0.5i;
X7(5) = 4;            % Nyquist (real)
X7(6) = conj(X7(4));
X7(7) = conj(X7(3));
X7(8) = conj(X7(2));
x7 = ifft(X7);
assert(isreal(x7));

% Test 8: ifft of conjugate-symmetric spectrum (odd length)
N8 = 7;
X8 = zeros(N8, 1);
X8(1) = 5;
X8(2) = 2 + 3i;
X8(3) = -1 - 2i;
X8(4) = 0.5 + 1i;
X8(5) = conj(X8(4));
X8(6) = conj(X8(3));
X8(7) = conj(X8(2));
x8 = ifft(X8);
assert(isreal(x8));

% Test 9: complex (non-real) input should NOT lose its imaginary part
x9 = randn(64, 1) + 1i * randn(64, 1);
X9 = fft(x9);
y9 = ifft(X9);
assert(~isreal(y9));
for k = 1:length(x9)
    assert(abs(y9(k) - x9(k)) < tol);
end

% Test 10: fft of real 2D matrix along dim 1 — each column's FFT
% should have conjugate symmetry
M = randn(64, 4);
FM = fft(M);
for col = 1:4
    assert(imag(FM(1, col)) == 0);
    for k = 2:32
        partner = 64 - k + 2;
        assert(real(FM(k, col)) == real(FM(partner, col)));
        assert(imag(FM(k, col)) == -imag(FM(partner, col)));
    end
end

disp('SUCCESS')
