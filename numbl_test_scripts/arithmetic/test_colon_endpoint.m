% Test colon operator endpoint precision
% MATLAB guarantees the last element equals the endpoint exactly

% 0:0.1:1 should end at exactly 1.0
x = 0:0.1:1;
assert(length(x) == 11, '0:0.1:1 should have 11 elements');
assert(x(end) == 1, '0:0.1:1 last element should be exactly 1');
assert(x(1) == 0, '0:0.1:1 first element should be exactly 0');

% 0:0.2:1 should end at exactly 1.0
y = 0:0.2:1;
assert(length(y) == 6, '0:0.2:1 should have 6 elements');
assert(y(end) == 1, '0:0.2:1 last element should be exactly 1');

% Negative step endpoint
z = 1:-0.1:0;
assert(length(z) == 11, '1:-0.1:0 should have 11 elements');
assert(z(end) == 0, '1:-0.1:0 last element should be exactly 0');
assert(z(1) == 1, '1:-0.1:0 first element should be exactly 1');

% 0:0.3:1 should not reach 1
w = 0:0.3:1;
assert(length(w) == 4, '0:0.3:1 should have 4 elements');
assert(abs(w(end) - 0.9) < 1e-10, '0:0.3:1 last element should be ~0.9');

disp('SUCCESS');
