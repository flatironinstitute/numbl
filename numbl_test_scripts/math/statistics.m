% Test statistical functions: mean, std, var

% Scalar mean
assert(mean([2, 4, 6]) == 4);

% Mean with dimension
A = [1, 2, 3; 5, 6, 7];
B = mean(A, 1);  % column-wise: [3, 4, 5]
assert(abs(B(1) - 3) < 1e-5);
assert(abs(B(2) - 4) < 1e-5);
assert(abs(B(3) - 5) < 1e-5);

C = mean(A, 2);  % row-wise: [2; 6]
assert(abs(C(1) - 2) < 1e-5);
assert(abs(C(2) - 6) < 1e-5);

% std: [0, 2, 4] → mean=2, ss=8, var=8/2=4, std=2
v2 = [0, 2, 4];
s = std(v2);
assert(abs(s - 2) < 1e-4);

% var
vr = var(v2);
assert(abs(vr - 4) < 1e-4);

disp('SUCCESS')
