% Test ellipj - Jacobi elliptic functions

% Basic scalar test
[s, c, d] = ellipj(0.5, 0.25);
assert(abs(s - 0.4751) < 1e-3, 'sn(0.5, 0.25)');
assert(abs(c - 0.8799) < 1e-3, 'cn(0.5, 0.25)');
assert(abs(d - 0.9714) < 1e-3, 'dn(0.5, 0.25)');

% Edge case: m = 0 (circular functions)
[s, c, d] = ellipj(1.0, 0);
assert(abs(s - sin(1.0)) < 1e-12, 'sn with m=0');
assert(abs(c - cos(1.0)) < 1e-12, 'cn with m=0');
assert(abs(d - 1.0) < 1e-12, 'dn with m=0');

% Edge case: m = 1 (hyperbolic functions)
[s, c, d] = ellipj(1.0, 1);
assert(abs(s - tanh(1.0)) < 1e-12, 'sn with m=1');
assert(abs(c - sech(1.0)) < 1e-12, 'cn with m=1');
assert(abs(d - sech(1.0)) < 1e-12, 'dn with m=1');

% Identity: sn^2 + cn^2 = 1
[s, c, d] = ellipj(2.3, 0.7);
assert(abs(s^2 + c^2 - 1) < 1e-12, 'sn^2 + cn^2 = 1');

% Identity: dn^2 + m*sn^2 = 1
assert(abs(d^2 + 0.7*s^2 - 1) < 1e-12, 'dn^2 + m*sn^2 = 1');

% u = 0
[s, c, d] = ellipj(0, 0.5);
assert(abs(s) < 1e-15, 'sn(0) = 0');
assert(abs(c - 1) < 1e-15, 'cn(0) = 1');
assert(abs(d - 1) < 1e-15, 'dn(0) = 1');

% Vector U, scalar M
U = [0, 0.5, 1.0];
[S, C, D] = ellipj(U, 0.25);
assert(length(S) == 3);
assert(length(C) == 3);
assert(length(D) == 3);
assert(abs(S(1)) < 1e-12, 'vector sn(0)');
assert(abs(S(2) - 0.4751) < 1e-3, 'vector sn(0.5)');

% Single output (just sn)
s = ellipj(0.5, 0.25);
assert(abs(s - 0.4751) < 1e-3, 'single output sn');

% Scalar U, vector M
M = [0, 0.5, 1.0];
[S, C, D] = ellipj(1.0, M);
assert(abs(S(1) - sin(1.0)) < 1e-12, 'scalar U, vector M: m=0');
assert(abs(S(3) - tanh(1.0)) < 1e-12, 'scalar U, vector M: m=1');

disp('SUCCESS');
