% Test JIT loop type inference: types assigned in the loop body must
% propagate back to the top of the loop (fixed-point iteration).

%% 2-variable chain (should work with 1 repass)
% a changes from number to complex inside the loop; b depends on a.
a = 1;
b = 1;
for k = 1:3
    b = a * 2;
    a = a * 1i;
end
% k=1: b = 1*2 = 2,    a = 1*1i = 1i
% k=2: b = 1i*2 = 2i,  a = 1i*1i = -1
% k=3: b = -1*2 = -2,  a = -1*1i = -1i
assert(abs(b - (-2)) < 1e-10, '2-var chain: b should be -2');
assert(abs(a - (-1i)) < 1e-10, '2-var chain: a should be -1i');

%% 3-variable chain (requires 2+ repasses to propagate correctly)
% a changes from number to complex, b depends on a, c depends on b.
% With only 1 repass, c's type won't be updated to complex.
a2 = 1;
b2 = 1;
c2 = 1;
for k = 1:3
    c2 = b2 * 2;
    b2 = a2 * 2;
    a2 = a2 * 1i;
end
% k=1: c2 = 1*2 = 2,   b2 = 1*2 = 2,   a2 = 1*1i = 1i
% k=2: c2 = 2*2 = 4,   b2 = 1i*2 = 2i, a2 = 1i*1i = -1
% k=3: c2 = 2i*2 = 4i, b2 = -1*2 = -2, a2 = -1*1i = -1i
assert(~isnan(real(c2)), '3-var chain: c2 real part should not be NaN');
assert(~isnan(imag(c2)), '3-var chain: c2 imag part should not be NaN');
assert(abs(c2 - 4i) < 1e-10, '3-var chain: c2 should be 4i');
assert(abs(b2 - (-2)) < 1e-10, '3-var chain: b2 should be -2');

%% 3-variable chain in while loop
a3 = 1;
b3 = 1;
c3 = 1;
k = 0;
while k < 3
    k = k + 1;
    c3 = b3 * 2;
    b3 = a3 * 2;
    a3 = a3 * 1i;
end
assert(abs(c3 - 4i) < 1e-10, '3-var while: c3 should be 4i');
assert(abs(b3 - (-2)) < 1e-10, '3-var while: b3 should be -2');

disp('SUCCESS');
