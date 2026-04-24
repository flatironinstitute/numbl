% LHS-on-RHS aliasing — `r` is read on the right and written on the left
% in the same statement. Under --opt e2, the kernel must allocate a
% fresh output buffer so reads of in_r see the OLD data while writes
% land in the freshly-allocated out_r.

n = 4000;
x = linspace(-1, 1, n);
y = linspace(0.1, 0.9, n);

r = x + y;
r = r - 0.5 .* x;
r = r .* y + 3.0;
r = r ./ (1 + abs(y));

s = sum(r);
% Verified in MATLAB R2025b.
expected = 9054.924255208036;
assert(abs(s - expected) < 1e-9, ...
    sprintf('aliasing: sum(r) = %.16g (expected %.16g)', s, expected));

disp('SUCCESS')
