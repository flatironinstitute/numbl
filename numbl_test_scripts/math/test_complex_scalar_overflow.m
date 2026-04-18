% Complex scalar ops must stay finite when re*re or im*im would overflow,
% and must stay accurate when both parts are tiny (underflow).  The fixes
% use hypot, Smith's algorithm (division), Kahan's algorithm (asin/acos),
% and LAPACK-style scaled accumulation (norm).

% --- sign ----------------------------------------------------------------
z = complex(1e200, 1e200);
s = sign(z);
assert(isfinite(real(s)) && isfinite(imag(s)), 'sign components must be finite');
expected = 1/sqrt(2);
assert(abs(real(s) - expected) < 1e-12, 'real(sign) wrong');
assert(abs(imag(s) - expected) < 1e-12, 'imag(sign) wrong');
assert(isequal(sign(complex(-1e200, 0)), -1), 'sign(-1e200+0i) must be -1');
z3 = complex(1e-200, 1e200);
s3 = sign(z3);
assert(abs(imag(s3) - 1) < 1e-12, 'imag(sign) ~ 1 for |im| >> |re|');
assert(isequal(sign(complex(0, 0)), 0), 'sign(0+0i) must be 0');

% --- sqrt ----------------------------------------------------------------
w = sqrt(complex(1e200, 1e200));
assert(isfinite(real(w)) && isfinite(imag(w)), 'sqrt components must be finite');
expected_mag = sqrt(sqrt(2)) * 1e100;
assert(abs(abs(w) - expected_mag) / expected_mag < 1e-12, 'sqrt magnitude wrong');

% --- scalar complex division --------------------------------------------
r1 = 1 ./ complex(1e200, 1e200);
assert(isfinite(real(r1)) && isfinite(imag(r1)), '1/z big must be finite');
exp1 = 1/(sqrt(2)*1e200);
assert(abs(abs(r1) - exp1)/exp1 < 1e-12, '|1/z_big| wrong');
r2 = 1 ./ complex(1e-200, 1e-200);
assert(isfinite(real(r2)) && isfinite(imag(r2)), '1/z tiny must be finite');
exp2 = 1/(sqrt(2)*1e-200);
assert(abs(abs(r2) - exp2)/exp2 < 1e-12, '|1/z_tiny| wrong');

% --- scalar inv of complex ----------------------------------------------
iv = inv(complex(1e200, 1e200));
assert(isfinite(real(iv)) && isfinite(imag(iv)), 'inv(complex) big must be finite');
assert(abs(abs(iv) - exp1)/exp1 < 1e-12, '|inv(z_big)| wrong');

% --- asin / acos of huge complex ----------------------------------------
za = complex(1e200, 1e200);
ra = asin(za);
assert(isfinite(real(ra)) && isfinite(imag(ra)), 'asin(big) finite');
assert(abs(real(ra) - pi/4) < 1e-12, 'Re(asin(big)) ~ pi/4');
% Im(asin(a*(1+i))) = log(2*sqrt(2)*a) for a large.
expected_im = log(2*sqrt(2)*1e200);
assert(abs(imag(ra) - expected_im)/expected_im < 1e-12, 'Im(asin(big)) wrong');

rb = acos(za);
assert(isfinite(real(rb)) && isfinite(imag(rb)), 'acos(big) finite');
assert(abs(real(rb) - pi/4) < 1e-12, 'Re(acos(big)) ~ pi/4');
assert(abs(imag(rb) + expected_im)/expected_im < 1e-12, 'Im(acos(big)) wrong');

% asin/acos branch-cut convention for real |x|>1 stays unchanged.
assert(abs(real(asin(2)) - pi/2) + abs(imag(asin(2)) + log(2+sqrt(3))) < 1e-12, 'asin(2) branch');
assert(abs(real(acos(2))) + abs(imag(acos(2)) - log(2+sqrt(3))) < 1e-12, 'acos(2) branch');
assert(abs(real(asin(-2)) + pi/2) + abs(imag(asin(-2)) - log(2+sqrt(3))) < 1e-12, 'asin(-2) branch');

% --- asec / acsc of tiny real (routes through recip + acos/asin) --------
as = asec(1e-200);
assert(isfinite(real(as)) && isfinite(imag(as)), 'asec(tiny) finite');
assert(abs(real(as)) < 1e-12, 'Re(asec(tiny)) ~ 0');
assert(abs(imag(as) - log(2*1e200))/log(2*1e200) < 1e-12, 'Im(asec(tiny)) ~ log(2/x)');

ac = acsc(1e-200);
assert(isfinite(real(ac)) && isfinite(imag(ac)), 'acsc(tiny) finite');
assert(abs(real(ac) - pi/2) < 1e-12, 'Re(acsc(tiny)) ~ pi/2');

% --- Frobenius / 2-norm with huge entries -------------------------------
A = [1e200, 1e200; 1e200, 1e200];
assert(isfinite(norm(A, 'fro')), 'norm fro must be finite');
assert(abs(norm(A, 'fro') - 2e200)/2e200 < 1e-12, 'norm fro value wrong');

v = [1e200; 1e200; 1e200];
assert(isfinite(norm(v)), 'vector 2-norm must be finite');
assert(abs(norm(v) - sqrt(3)*1e200)/(sqrt(3)*1e200) < 1e-12, 'vector 2-norm wrong');

disp('SUCCESS');
