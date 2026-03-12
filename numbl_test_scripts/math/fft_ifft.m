% Test fft and ifft against brute-force DFT computation

tol = 1e-8;

% Test 1: fft of real column vector
x1 = [1; 2; 3; 4];
X1 = fft(x1);
X1_bf = brute_dft(x1);
for k = 1:length(x1)
    assert(abs(X1(k) - X1_bf(k)) < tol);
end

% Test 2: fft of real row vector (output should also be row)
x2 = [1, 2, 3, 4, 5, 6, 7, 8];
X2 = fft(x2);
X2_bf = brute_dft(x2.');
assert(size(X2, 1) == 1);
assert(size(X2, 2) == 8);
for k = 1:length(x2)
    assert(abs(X2(k) - X2_bf(k)) < tol);
end

% Test 3: fft of complex column vector
x3 = [1+2i; 3-1i; 2+0i; 0+1i];
X3 = fft(x3);
X3_bf = brute_dft(x3);
for k = 1:length(x3)
    assert(abs(X3(k) - X3_bf(k)) < tol);
end

% Test 4: ifft of complex vector
X4 = [10+0i; -2+2i; -2+0i; -2-2i];
x4 = ifft(X4);
x4_bf = brute_idft(X4);
for k = 1:length(X4)
    assert(abs(x4(k) - x4_bf(k)) < tol);
end

% Test 5: ifft(fft(x)) should recover x
x5 = [3; 1; 4; 1; 5; 9; 2; 6];
x5_rt = ifft(fft(x5));
for k = 1:length(x5)
    assert(abs(x5_rt(k) - x5(k)) < tol);
end

% Test 6: fft(ifft(X)) should recover X
X6 = [10+0i; -2+2i; -2+0i; -2-2i; 1+0i; 0+3i; -1-1i; 3+0i];
X6_rt = fft(ifft(X6));
for k = 1:length(X6)
    assert(abs(X6_rt(k) - X6(k)) < tol);
end

% Test 7: fft with n (zero-padding)
x7 = [1; 2; 3; 4];
X7 = fft(x7, 8);
x7_pad = [1; 2; 3; 4; 0; 0; 0; 0];
X7_bf = brute_dft(x7_pad);
assert(length(X7) == 8);
for k = 1:8
    assert(abs(X7(k) - X7_bf(k)) < tol);
end

disp('SUCCESS')

function X = brute_dft(x)
    % Brute-force DFT: X[k] = sum_{n=0}^{N-1} x[n]*exp(-2*pi*i*n*k/N)
    N = length(x);
    X = zeros(N, 1);
    for k = 1:N
        s = 0;
        for n = 1:N
            s = s + x(n) * exp(-2*pi*1i*(n-1)*(k-1)/N);
        end
        X(k) = s;
    end
end

function x = brute_idft(X)
    % Brute-force IDFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k]*exp(2*pi*i*k*n/N)
    N = length(X);
    x = zeros(N, 1);
    for n = 1:N
        s = 0;
        for k = 1:N
            s = s + X(k) * exp(2*pi*1i*(k-1)*(n-1)/N);
        end
        x(n) = s / N;
    end
end
