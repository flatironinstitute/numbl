% Test isnan, isinf, isfinite on complex numbers

% isnan on complex scalars
assert(isnan(NaN + 1i));
assert(isnan(1 + NaN*1i));
assert(~isnan(1 + 2i));
assert(~isnan(3 + 0i));

% isinf on complex scalars
assert(isinf(Inf + 1i));
assert(isinf(1 + Inf*1i));
assert(~isinf(1 + 2i));
assert(~isinf(3 + 0i));

% isfinite on complex scalars
assert(isfinite(1 + 2i));
assert(isfinite(3 + 0i));
assert(~isfinite(Inf + 1i));
assert(~isfinite(1 + Inf*1i));
assert(~isfinite(NaN + 1i));

% isnan on complex vectors
v = [1+2i, NaN+1i, 3+0i, 1+NaN*1i];
r = isnan(v);
assert(r(1) == 0);
assert(r(2) == 1);
assert(r(3) == 0);
assert(r(4) == 1);

% isinf on complex vectors
v2 = [1+2i, Inf+1i, 3+0i, 1+Inf*1i];
r2 = isinf(v2);
assert(r2(1) == 0);
assert(r2(2) == 1);
assert(r2(3) == 0);
assert(r2(4) == 1);

% isfinite on complex vectors
v3 = [1+2i, Inf+1i, NaN+0i, 3+4i];
r3 = isfinite(v3);
assert(r3(1) == 1);
assert(r3(2) == 0);
assert(r3(3) == 0);
assert(r3(4) == 1);

disp('SUCCESS');
