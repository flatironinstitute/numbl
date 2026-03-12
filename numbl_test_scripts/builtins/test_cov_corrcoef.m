% Test cov and corrcoef builtins

% Variance of a vector (cov of single vector)
x = [1 2 3 4 5];
c = cov(x);
assert(abs(c - 2.5) < 1e-10);

% Covariance matrix of two vectors
x2 = [1 2 3 4 5];
y2 = [2 4 6 8 10];
C = cov(x2, y2);
assert(size(C, 1) == 2);
assert(size(C, 2) == 2);
assert(abs(C(1,1) - 2.5) < 1e-10);
assert(abs(C(1,2) - 5) < 1e-10);
assert(abs(C(2,1) - 5) < 1e-10);
assert(abs(C(2,2) - 10) < 1e-10);

% Covariance of a matrix (columns as variables)
M = [1 2; 3 4; 5 6];
Cm = cov(M);
assert(size(Cm, 1) == 2);
assert(size(Cm, 2) == 2);
assert(abs(Cm(1,1) - 4) < 1e-10);
assert(abs(Cm(1,2) - 4) < 1e-10);
assert(abs(Cm(2,1) - 4) < 1e-10);
assert(abs(Cm(2,2) - 4) < 1e-10);

% corrcoef - perfect positive correlation
R = corrcoef(x2, y2);
assert(size(R, 1) == 2);
assert(size(R, 2) == 2);
assert(abs(R(1,1) - 1) < 1e-10);
assert(abs(R(1,2) - 1) < 1e-10);
assert(abs(R(2,1) - 1) < 1e-10);
assert(abs(R(2,2) - 1) < 1e-10);

% corrcoef - negative correlation
y_neg = [10 8 6 4 2];
R2 = corrcoef(x2, y_neg);
assert(abs(R2(1,2) - (-1)) < 1e-10);

% corrcoef of a matrix
R3 = corrcoef(M);
assert(abs(R3(1,1) - 1) < 1e-10);
assert(abs(R3(1,2) - 1) < 1e-10);

disp('SUCCESS');
