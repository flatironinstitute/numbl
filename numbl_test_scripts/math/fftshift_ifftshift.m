% Test fftshift and ifftshift

tol = 1e-10;

% Test 1: fftshift of column vector (even length)
% floor(6/2)=3, circshift by 3: [4;5;6;1;2;3]
x1 = [1; 2; 3; 4; 5; 6];
y1 = fftshift(x1);
assert(y1(1) == 4);
assert(y1(2) == 5);
assert(y1(3) == 6);
assert(y1(4) == 1);
assert(y1(5) == 2);
assert(y1(6) == 3);

% Test 2: fftshift of column vector (odd length)
% floor(5/2)=2, circshift by 2: [4;5;1;2;3]
x2 = [1; 2; 3; 4; 5];
y2 = fftshift(x2);
assert(y2(1) == 4);
assert(y2(2) == 5);
assert(y2(3) == 1);
assert(y2(4) == 2);
assert(y2(5) == 3);

% Test 3: ifftshift reverses fftshift (even length)
x3 = [1; 2; 3; 4; 5; 6];
y3 = ifftshift(fftshift(x3));
for k = 1:length(x3)
    assert(abs(y3(k) - x3(k)) < tol);
end

% Test 4: ifftshift reverses fftshift (odd length)
x4 = [1; 2; 3; 4; 5];
y4 = ifftshift(fftshift(x4));
for k = 1:length(x4)
    assert(abs(y4(k) - x4(k)) < tol);
end

% Test 5: fftshift of row vector (even length)
x5 = [1 2 3 4];
y5 = fftshift(x5);
assert(y5(1) == 3);
assert(y5(2) == 4);
assert(y5(3) == 1);
assert(y5(4) == 2);
assert(size(y5, 1) == 1);
assert(size(y5, 2) == 4);

% Test 6: fftshift of matrix (all dims)
% A is 4x2; dim1 shift by 2, dim2 shift by 1
A = [1 2; 3 4; 5 6; 7 8];
Y6 = fftshift(A);
assert(Y6(1,1) == 6);
assert(Y6(1,2) == 5);
assert(Y6(2,1) == 8);
assert(Y6(2,2) == 7);
assert(Y6(3,1) == 2);
assert(Y6(3,2) == 1);
assert(Y6(4,1) == 4);
assert(Y6(4,2) == 3);

% Test 7: fftshift with dim=1 (rows only)
Y7 = fftshift(A, 1);
assert(Y7(1,1) == 5);
assert(Y7(1,2) == 6);
assert(Y7(2,1) == 7);
assert(Y7(2,2) == 8);
assert(Y7(3,1) == 1);
assert(Y7(3,2) == 2);
assert(Y7(4,1) == 3);
assert(Y7(4,2) == 4);

% Test 8: fftshift with dim=2 (columns only)
Y8 = fftshift(A, 2);
assert(Y8(1,1) == 2);
assert(Y8(1,2) == 1);
assert(Y8(2,1) == 4);
assert(Y8(2,2) == 3);

% Test 9: ifftshift with dim parameter
Y9 = ifftshift(fftshift(A, 1), 1);
for r = 1:4
    for c = 1:2
        assert(abs(Y9(r,c) - A(r,c)) < tol);
    end
end

% Test 10: fft then fftshift round-trip via ifftshift
x10 = [1; 2; 3; 4; 5; 6; 7; 8];
X10 = fftshift(fft(x10));
x10r = ifft(ifftshift(X10));
for k = 1:length(x10)
    assert(abs(real(x10r(k)) - x10(k)) < tol);
    assert(abs(imag(x10r(k))) < tol);
end

% Test 11: 3D array fftshift along dim=2
B = reshape(1:24, 2, 3, 4);
YB = fftshift(B, 2);
% Along dim=2 (size=3, odd): shift by floor(3/2)=1
% Original col order: 1,2,3 → shifted to: 3,1,2
assert(YB(1,1,1) == B(1,3,1));
assert(YB(1,2,1) == B(1,1,1));
assert(YB(1,3,1) == B(1,2,1));

disp('SUCCESS')
