% Test fft/ifft with N-dimensional inputs and dim parameter

tol = 1e-8;

% Test 1: fft of 2D matrix - column-wise (default dim=1)
A = [1 2; 3 4; 5 6];
Y = fft(A);
y1_bf = brute_dft([1; 3; 5]);
y2_bf = brute_dft([2; 4; 6]);
assert(size(Y, 1) == 3);
assert(size(Y, 2) == 2);
for k = 1:3
    assert(abs(Y(k,1) - y1_bf(k)) < tol);
    assert(abs(Y(k,2) - y2_bf(k)) < tol);
end

% Test 2: fft along dim=2 (row-wise)
Y2 = fft(A, [], 2);
y_r1_bf = brute_dft([1; 2]);
y_r2_bf = brute_dft([3; 4]);
y_r3_bf = brute_dft([5; 6]);
assert(size(Y2, 1) == 3);
assert(size(Y2, 2) == 2);
for k = 1:2
    assert(abs(Y2(1,k) - y_r1_bf(k)) < tol);
    assert(abs(Y2(2,k) - y_r2_bf(k)) < tol);
    assert(abs(Y2(3,k) - y_r3_bf(k)) < tol);
end

% Test 3: fft with n (zero-pad) column-wise
Y3 = fft(A, 4);
y3c1_bf = brute_dft([1; 3; 5; 0]);
y3c2_bf = brute_dft([2; 4; 6; 0]);
assert(size(Y3, 1) == 4);
assert(size(Y3, 2) == 2);
for k = 1:4
    assert(abs(Y3(k,1) - y3c1_bf(k)) < tol);
    assert(abs(Y3(k,2) - y3c2_bf(k)) < tol);
end

% Test 4: fft with n and dim=2 (zero-pad rows to length 4)
Y4 = fft(A, 4, 2);
y4r1_bf = brute_dft([1; 2; 0; 0]);
y4r2_bf = brute_dft([3; 4; 0; 0]);
assert(size(Y4, 1) == 3);
assert(size(Y4, 2) == 4);
for k = 1:4
    assert(abs(Y4(1,k) - y4r1_bf(k)) < tol);
    assert(abs(Y4(2,k) - y4r2_bf(k)) < tol);
end

% Test 5: row vector default dim (first non-singleton = dim 2)
x5 = [1 2 3 4];
Y5 = fft(x5);
Y5_bf = brute_dft(x5.');
assert(size(Y5, 1) == 1);
assert(size(Y5, 2) == 4);
for k = 1:4
    assert(abs(Y5(k) - Y5_bf(k)) < tol);
end

% Test 6: 3D array, fft along dim=1
B = reshape(1:24, 2, 3, 4);
YB1 = fft(B, [], 1);
% Check one fiber: B(:,2,3) = [B(1,2,3); B(2,2,3)] = [15; 16]
fiber_b23 = [B(1,2,3); B(2,2,3)];
bf_b23 = brute_dft(fiber_b23);
assert(size(YB1, 1) == 2);
assert(size(YB1, 2) == 3);
assert(size(YB1, 3) == 4);
for k = 1:2
    assert(abs(YB1(k,2,3) - bf_b23(k)) < tol);
end

% Test 7: 3D array, fft along dim=2
YB2 = fft(B, [], 2);
% Fiber B(1,:,2) = [B(1,1,2); B(1,2,2); B(1,3,2)] = [7; 9; 11]
fiber_1x2 = [B(1,1,2); B(1,2,2); B(1,3,2)];
bf_1x2 = brute_dft(fiber_1x2);
assert(size(YB2, 1) == 2);
assert(size(YB2, 2) == 3);
assert(size(YB2, 3) == 4);
for k = 1:3
    assert(abs(YB2(1,k,2) - bf_1x2(k)) < tol);
end

% Test 8: 3D array, fft along dim=3
YB3 = fft(B, [], 3);
% Fiber B(2,1,:) = [2; 8; 14; 20]
fiber_21 = [B(2,1,1); B(2,1,2); B(2,1,3); B(2,1,4)];
bf_21 = brute_dft(fiber_21);
assert(size(YB3, 1) == 2);
assert(size(YB3, 2) == 3);
assert(size(YB3, 3) == 4);
for k = 1:4
    assert(abs(YB3(2,1,k) - bf_21(k)) < tol);
end

% Test 9: ifft along dim=2
Y9 = fft(A, [], 2);
A9 = ifft(Y9, [], 2);
for r = 1:3
    for c = 1:2
        assert(abs(A9(r,c) - A(r,c)) < tol);
    end
end

disp('SUCCESS')

function X = brute_dft(x)
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
