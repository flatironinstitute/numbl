% Test sort with complex numbers
% MATLAB sorts complex by magnitude, ties by phase angle

% Vector sort by magnitude
v = [3+4i, 1+0i, 0+2i];  % magnitudes: 5, 1, 2
s = sort(v);
assert(abs(s(1) - (1+0i)) < 1e-10);
assert(abs(s(2) - (0+2i)) < 1e-10);
assert(abs(s(3) - (3+4i)) < 1e-10);

% Sort descending
sd = sort(v, 'descend');
assert(abs(sd(1) - (3+4i)) < 1e-10);
assert(abs(sd(2) - (0+2i)) < 1e-10);
assert(abs(sd(3) - (1+0i)) < 1e-10);

% Sort with index output
[s2, idx] = sort(v);
assert(idx(1) == 2);
assert(idx(2) == 3);
assert(idx(3) == 1);

% Sort scalar complex
z = 3+4i;
sz = sort(z);
assert(abs(sz - z) < 1e-10);

% Tie-breaking by phase angle (all magnitude 1)
v3 = [0+1i, 1+0i, -1+0i];  % phases: pi/2, 0, pi
s3 = sort(v3);
assert(abs(s3(1) - (1+0i)) < 1e-10);   % phase 0 is smallest
assert(abs(s3(2) - (0+1i)) < 1e-10);   % phase pi/2
assert(abs(s3(3) - (-1+0i)) < 1e-10);  % phase pi is largest

disp('SUCCESS')
