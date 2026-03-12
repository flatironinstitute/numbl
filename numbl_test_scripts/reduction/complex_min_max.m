% Test min and max with complex numbers
% MATLAB compares by real part, ties broken by imaginary part

% Basic min/max
m1 = min([1+2i, 3+4i]);
assert(abs(m1 - (1+2i)) < 1e-10);
m2 = max([1+2i, 3+4i]);
assert(abs(m2 - (3+4i)) < 1e-10);

% Tie on real part, compare imaginary
m3 = min([1+3i, 1+1i]);
assert(abs(m3 - (1+1i)) < 1e-10);
m4 = max([1+3i, 1+1i]);
assert(abs(m4 - (1+3i)) < 1e-10);

% With index output
[m5, i5] = min([3+4i, 1+2i]);
assert(abs(m5 - (1+2i)) < 1e-10);
assert(i5 == 2);
[m6, i6] = max([3+4i, 1+2i]);
assert(abs(m6 - (3+4i)) < 1e-10);
assert(i6 == 1);

% Two-arg element-wise min/max
m7 = min(1+2i, 3+4i);
assert(abs(m7 - (1+2i)) < 1e-10);
m8 = max(1+2i, 3+4i);
assert(abs(m8 - (3+4i)) < 1e-10);

% Scalar complex input
m9 = min(3+4i);
assert(abs(m9 - (3+4i)) < 1e-10);
m10 = max(3+4i);
assert(abs(m10 - (3+4i)) < 1e-10);

% Column vector
v = [3+4i; 1+2i];
mv = min(v);
assert(abs(mv - (1+2i)) < 1e-10);

% Matrix (reduces along dim 1)
M = [3+4i, 1+0i; 1+2i, 5+6i];
mm = min(M);
assert(abs(mm(1) - (1+2i)) < 1e-10);
assert(abs(mm(2) - (1+0i)) < 1e-10);

disp('SUCCESS')
