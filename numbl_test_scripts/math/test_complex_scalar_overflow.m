% Complex scalar ops must stay finite when re*re or im*im would overflow,
% and must stay accurate when both parts are tiny (underflow).  Root fix
% is hypot/Smith's algorithm in sign, sqrt, scalar division, and scalar inv.

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

% --- scalar inv of complex ---------------------------------------------
iv = inv(complex(1e200, 1e200));
assert(isfinite(real(iv)) && isfinite(imag(iv)), 'inv(complex) big must be finite');
assert(abs(abs(iv) - exp1)/exp1 < 1e-12, '|inv(z_big)| wrong');

disp('SUCCESS');
