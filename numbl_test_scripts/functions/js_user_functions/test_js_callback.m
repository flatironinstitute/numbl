% Test passing a numbl function handle into a .numbl.js user function,
% which integrates it (here via the JS fallback; with myquad.wasm present
% the same handle is called back from WASM through numbl_cb_d).

% Anonymous handle: integral of x^2 over [0,1] = 1/3.
r1 = myquad(@(x) x.^2, 0, 1);
assert(abs(r1 - 1/3) < 1e-4, 'Expected myquad(@(x) x.^2, 0, 1) ~= 1/3');

% Named builtin handle: integral of sin over [0,pi] = 2.
r2 = myquad(@sin, 0, pi);
assert(abs(r2 - 2) < 1e-4, 'Expected myquad(@sin, 0, pi) ~= 2');

% Closure capturing a workspace variable.
k = 3;
r3 = myquad(@(x) k * x, 0, 2);   % k * x integrated over [0,2] = k*2 = 6
assert(abs(r3 - 6) < 1e-4, 'Expected myquad(@(x) k*x, 0, 2) ~= 6');

disp('SUCCESS')
