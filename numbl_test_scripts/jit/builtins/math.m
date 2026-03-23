% ===== sin =====

function test_sin_scalar()
assert(isequal(sin(0), 0));
assert(abs(sin(pi/2) - 1) < 1e-15);
assert(abs(sin(pi)) < 1e-15);
assert(abs(sin(-pi/2) + 1) < 1e-15);
assert(abs(sin(pi/6) - 0.5) < 1e-15);
end

function test_sin_complex()
% sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
z = 1 + 2i;
r = sin(z);
assert(abs(real(r) - sin(1)*cosh(2)) < 1e-12);
assert(abs(imag(r) - cos(1)*sinh(2)) < 1e-12);
assert(abs(sin(1i) - 1i*sinh(1)) < 1e-12);
end

function test_sin_tensor()
assert(isequal(sin([0 0 0]), [0 0 0]));
x = [0 pi/2 pi];
r = sin(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - 1) < 1e-15);
assert(abs(r(3)) < 1e-15);
% 2D tensor
x2 = [0 pi/2; pi 3*pi/2];
r2 = sin(x2);
assert(abs(r2(1,1)) < 1e-15);
assert(abs(r2(1,2) - 1) < 1e-15);
end

function test_sin_complex_tensor()
z = [1+2i 0+1i];
r = sin(z);
assert(abs(real(r(1)) - sin(1)*cosh(2)) < 1e-12);
assert(abs(imag(r(2)) - sinh(1)) < 1e-12);
end

% ===== cos =====

function test_cos_scalar()
assert(isequal(cos(0), 1));
assert(abs(cos(pi/2)) < 1e-15);
assert(abs(cos(pi) + 1) < 1e-15);
assert(abs(cos(pi/3) - 0.5) < 1e-15);
end

function test_cos_complex()
z = 1 + 2i;
r = cos(z);
assert(abs(real(r) - cos(1)*cosh(2)) < 1e-12);
assert(abs(imag(r) + sin(1)*sinh(2)) < 1e-12);
end

function test_cos_tensor()
x = [0 pi/2 pi];
r = cos(x);
assert(abs(r(1) - 1) < 1e-15);
assert(abs(r(2)) < 1e-15);
assert(abs(r(3) + 1) < 1e-15);
end

function test_cos_complex_tensor()
z = [1+2i 0+0i];
r = cos(z);
assert(abs(real(r(1)) - cos(1)*cosh(2)) < 1e-12);
assert(abs(r(2) - 1) < 1e-12);
end

% ===== tan =====

function test_tan_scalar()
assert(isequal(tan(0), 0));
assert(abs(tan(pi/4) - 1) < 1e-15);
assert(abs(tan(-pi/4) + 1) < 1e-15);
end

function test_tan_complex()
z = 1 + 1i;
r = tan(z);
expected_re = sin(2*1)/(cos(2*1) + cosh(2*1));
expected_im = sinh(2*1)/(cos(2*1) + cosh(2*1));
assert(abs(real(r) - expected_re) < 1e-12);
assert(abs(imag(r) - expected_im) < 1e-12);
end

function test_tan_tensor()
x = [0 pi/4 -pi/4];
r = tan(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - 1) < 1e-15);
assert(abs(r(3) + 1) < 1e-15);
end

% ===== asin =====

function test_asin_scalar()
assert(isequal(asin(0), 0));
assert(abs(asin(1) - pi/2) < 1e-15);
assert(abs(asin(-1) + pi/2) < 1e-15);
assert(abs(asin(0.5) - pi/6) < 1e-15);
end

function test_asin_complex()
% asin(2) should produce complex result
r = asin(2);
assert(abs(real(r) - pi/2) < 1e-12);
assert(imag(r) ~= 0);
% asin of complex input
z = 1 + 1i;
r = asin(z);
assert(abs(sin(r) - z) < 1e-10);
end

function test_asin_tensor()
x = [0 0.5 1];
r = asin(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - pi/6) < 1e-15);
assert(abs(r(3) - pi/2) < 1e-15);
end

% ===== acos =====

function test_acos_scalar()
assert(abs(acos(1)) < 1e-15);
assert(abs(acos(0) - pi/2) < 1e-15);
assert(abs(acos(-1) - pi) < 1e-15);
assert(abs(acos(0.5) - pi/3) < 1e-15);
end

function test_acos_complex()
% acos(2) should produce complex result
r = acos(2);
assert(abs(real(r)) < 1e-12);
assert(imag(r) ~= 0);
end

function test_acos_tensor()
x = [1 0 -1];
r = acos(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - pi/2) < 1e-15);
assert(abs(r(3) - pi) < 1e-15);
end

% ===== atan =====

function test_atan_scalar()
assert(isequal(atan(0), 0));
assert(abs(atan(1) - pi/4) < 1e-15);
assert(abs(atan(-1) + pi/4) < 1e-15);
end

function test_atan_complex()
z = 0 + 0.5i;
r = atan(z);
% verify: tan(atan(z)) == z
assert(abs(tan(r) - z) < 1e-10);
end

function test_atan_tensor()
x = [0 1 -1];
r = atan(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - pi/4) < 1e-15);
assert(abs(r(3) + pi/4) < 1e-15);
end

% ===== sinh =====

function test_sinh_scalar()
assert(isequal(sinh(0), 0));
assert(abs(sinh(1) - (exp(1) - exp(-1))/2) < 1e-15);
end

function test_sinh_complex()
z = 1 + 2i;
r = sinh(z);
assert(abs(real(r) - sinh(1)*cos(2)) < 1e-12);
assert(abs(imag(r) - cosh(1)*sin(2)) < 1e-12);
end

function test_sinh_tensor()
x = [0 1 -1];
r = sinh(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - sinh(1)) < 1e-15);
assert(abs(r(3) + sinh(1)) < 1e-15);
end

% ===== cosh =====

function test_cosh_scalar()
assert(isequal(cosh(0), 1));
assert(abs(cosh(1) - (exp(1) + exp(-1))/2) < 1e-15);
end

function test_cosh_complex()
z = 1 + 2i;
r = cosh(z);
assert(abs(real(r) - cosh(1)*cos(2)) < 1e-12);
assert(abs(imag(r) - sinh(1)*sin(2)) < 1e-12);
end

function test_cosh_tensor()
x = [0 1 -1];
r = cosh(x);
assert(abs(r(1) - 1) < 1e-15);
assert(abs(r(2) - cosh(1)) < 1e-15);
assert(abs(r(3) - cosh(1)) < 1e-15);
end

% ===== tanh =====

function test_tanh_scalar()
assert(isequal(tanh(0), 0));
assert(abs(tanh(1) - (exp(2) - 1)/(exp(2) + 1)) < 1e-15);
end

function test_tanh_complex()
z = 1 + 1i;
r = tanh(z);
expected_re = sinh(2*1)/(cosh(2*1) + cos(2*1));
expected_im = sin(2*1)/(cosh(2*1) + cos(2*1));
assert(abs(real(r) - expected_re) < 1e-12);
assert(abs(imag(r) - expected_im) < 1e-12);
end

function test_tanh_tensor()
x = [0 1 -1];
r = tanh(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - tanh(1)) < 1e-15);
assert(abs(r(3) + tanh(1)) < 1e-15);
end

% ===== exp =====

function test_exp_scalar()
assert(isequal(exp(0), 1));
assert(abs(exp(1) - exp(1)) < 1e-15);
assert(abs(exp(-1) - 1/exp(1)) < 1e-15);
end

function test_exp_complex()
% exp(i*pi) = -1 (Euler's formula)
r = exp(1i * pi);
assert(abs(real(r) + 1) < 1e-14);
assert(abs(imag(r)) < 1e-14);
% exp(1+2i) = exp(1)*(cos(2) + i*sin(2))
z = 1 + 2i;
r = exp(z);
assert(abs(real(r) - exp(1)*cos(2)) < 1e-12);
assert(abs(imag(r) - exp(1)*sin(2)) < 1e-12);
end

function test_exp_tensor()
x = [0 1 -1];
r = exp(x);
assert(abs(r(1) - 1) < 1e-15);
assert(abs(r(2) - exp(1)) < 1e-15);
assert(abs(r(3) - exp(-1)) < 1e-15);
end

function test_exp_complex_tensor()
z = [0+0i 1i*pi];
r = exp(z);
assert(abs(r(1) - 1) < 1e-12);
assert(abs(real(r(2)) + 1) < 1e-12);
end

% ===== log =====

function test_log_scalar()
assert(isequal(log(1), 0));
assert(abs(log(exp(1)) - 1) < 1e-15);
assert(abs(log(exp(3)) - 3) < 1e-15);
end

function test_log_complex()
% log(-1) = i*pi
r = log(-1 + 0i);
assert(abs(real(r)) < 1e-14);
assert(abs(imag(r) - pi) < 1e-14);
% log(1i) = i*pi/2
r = log(1i);
assert(abs(real(r)) < 1e-14);
assert(abs(imag(r) - pi/2) < 1e-14);
end

function test_log_tensor()
x = [1 exp(1) exp(2)];
r = log(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - 1) < 1e-15);
assert(abs(r(3) - 2) < 1e-15);
end

function test_log_complex_tensor()
z = [1+0i -1+0i];
r = log(z);
assert(abs(r(1)) < 1e-12);
assert(abs(real(r(2))) < 1e-12);
assert(abs(imag(r(2)) - pi) < 1e-12);
end

% ===== log2 =====

function test_log2_scalar()
assert(isequal(log2(1), 0));
assert(isequal(log2(2), 1));
assert(isequal(log2(4), 2));
assert(isequal(log2(8), 3));
assert(abs(log2(1024) - 10) < 1e-12);
end

function test_log2_complex()
r = log2(1 + 1i);
expected = log(1+1i) / log(2);
assert(abs(real(r) - real(expected)) < 1e-12);
assert(abs(imag(r) - imag(expected)) < 1e-12);
end

function test_log2_tensor()
x = [1 2 4 8];
r = log2(x);
assert(isequal(r, [0 1 2 3]));
end

function test_log2_frexp()
% [f, e] = log2(x) gives f * 2^e = x, 0.5 <= |f| < 1
[f, e] = log2(8);
assert(abs(f - 0.5) < 1e-15);
assert(isequal(e, 4));
assert(abs(f * 2^e - 8) < 1e-15);
[f, e] = log2(0);
assert(isequal(f, 0));
assert(isequal(e, 0));
end

function test_log2_frexp_tensor()
[f, e] = log2([8 16 0.5]);
assert(abs(f(1) - 0.5) < 1e-15);
assert(isequal(e(1), 4));
assert(abs(f(2) - 0.5) < 1e-15);
assert(isequal(e(2), 5));
assert(abs(f(3) - 0.5) < 1e-15);
assert(isequal(e(3), 0));
end

% ===== log10 =====

function test_log10_scalar()
assert(isequal(log10(1), 0));
assert(isequal(log10(10), 1));
assert(isequal(log10(100), 2));
assert(abs(log10(1000) - 3) < 1e-12);
end

function test_log10_complex()
r = log10(1 + 1i);
expected = log(1+1i) / log(10);
assert(abs(real(r) - real(expected)) < 1e-12);
assert(abs(imag(r) - imag(expected)) < 1e-12);
end

function test_log10_tensor()
x = [1 10 100 1000];
r = log10(x);
assert(abs(r(1)) < 1e-15);
assert(abs(r(2) - 1) < 1e-15);
assert(abs(r(3) - 2) < 1e-15);
assert(abs(r(4) - 3) < 1e-12);
end

% ===== abs =====

function test_abs_scalar()
assert(isequal(abs(5), 5));
assert(isequal(abs(-3), 3));
assert(isequal(abs(0), 0));
end

function test_abs_complex()
assert(isequal(abs(3 + 4i), 5));
assert(isequal(abs(-3 - 4i), 5));
assert(abs(abs(1 + 1i) - sqrt(2)) < 1e-15);
end

function test_abs_tensor()
x = [1 -2 3 -4];
assert(isequal(abs(x), [1 2 3 4]));
x2 = [-1 2; -3 4];
assert(isequal(abs(x2), [1 2; 3 4]));
end

function test_abs_complex_tensor()
z = [3+4i -3-4i];
r = abs(z);
assert(isequal(r, [5 5]));
end

% ===== sqrt =====

function test_sqrt_scalar()
assert(isequal(sqrt(4), 2));
assert(isequal(sqrt(9), 3));
assert(isequal(sqrt(0), 0));
assert(isequal(sqrt(1), 1));
assert(abs(sqrt(2) - 1.41421356237310) < 1e-12);
end

function test_sqrt_complex()
% sqrt(-1) = i
r = sqrt(-1);
assert(abs(real(r)) < 1e-15);
assert(abs(imag(r) - 1) < 1e-15);
% sqrt(-4) = 2i
r = sqrt(-4);
assert(abs(real(r)) < 1e-15);
assert(abs(imag(r) - 2) < 1e-15);
% sqrt(1i) = (1+i)/sqrt(2)
r = sqrt(1i);
assert(abs(real(r) - 1/sqrt(2)) < 1e-12);
assert(abs(imag(r) - 1/sqrt(2)) < 1e-12);
end

function test_sqrt_tensor()
x = [0 1 4 9 16];
assert(isequal(sqrt(x), [0 1 2 3 4]));
end

function test_sqrt_complex_tensor()
z = [4+0i -1+0i];
r = sqrt(z);
assert(abs(r(1) - 2) < 1e-12);
assert(abs(real(r(2))) < 1e-12);
assert(abs(imag(r(2)) - 1) < 1e-12);
end

% ===== sign =====

function test_sign_scalar()
assert(isequal(sign(5), 1));
assert(isequal(sign(-3), -1));
assert(isequal(sign(0), 0));
end

function test_sign_complex()
% sign(z) = z / abs(z) for z ~= 0
z = 3 + 4i;
r = sign(z);
assert(abs(real(r) - 3/5) < 1e-15);
assert(abs(imag(r) - 4/5) < 1e-15);
assert(isequal(sign(0+0i), 0));
end

function test_sign_tensor()
x = [5 -3 0 2 -1];
assert(isequal(sign(x), [1 -1 0 1 -1]));
end

function test_sign_complex_tensor()
z = [3+4i -3-4i 0+0i];
r = sign(z);
assert(abs(real(r(1)) - 3/5) < 1e-12);
assert(abs(imag(r(1)) - 4/5) < 1e-12);
assert(abs(real(r(2)) + 3/5) < 1e-12);
assert(abs(imag(r(2)) + 4/5) < 1e-12);
assert(abs(r(3)) < 1e-12);
end

% ===== floor =====

function test_floor_scalar()
assert(isequal(floor(3.7), 3));
assert(isequal(floor(-3.2), -4));
assert(isequal(floor(5), 5));
assert(isequal(floor(0), 0));
end

function test_floor_complex()
r = floor(3.7 + 2.3i);
assert(isequal(real(r), 3));
assert(isequal(imag(r), 2));
end

function test_floor_tensor()
x = [1.2 2.8 -0.5 3.0];
assert(isequal(floor(x), [1 2 -1 3]));
end

% ===== ceil =====

function test_ceil_scalar()
assert(isequal(ceil(3.2), 4));
assert(isequal(ceil(-3.7), -3));
assert(isequal(ceil(5), 5));
assert(isequal(ceil(0), 0));
end

function test_ceil_complex()
r = ceil(3.2 + 2.7i);
assert(isequal(real(r), 4));
assert(isequal(imag(r), 3));
end

function test_ceil_tensor()
x = [1.2 2.8 -0.5 3.0];
assert(isequal(ceil(x), [2 3 0 3]));
end

% ===== fix =====

function test_fix_scalar()
assert(isequal(fix(3.7), 3));
assert(isequal(fix(-3.7), -3));
assert(isequal(fix(5), 5));
assert(isequal(fix(0), 0));
end

function test_fix_complex()
r = fix(3.7 + 2.3i);
assert(isequal(real(r), 3));
assert(isequal(imag(r), 2));
end

function test_fix_tensor()
x = [1.7 -2.3 0.9 -0.9];
assert(isequal(fix(x), [1 -2 0 0]));
end

% ===== round =====

function test_round_scalar()
assert(isequal(round(3.4), 3));
assert(isequal(round(3.5), 4));
assert(isequal(round(-3.5), -4));
assert(isequal(round(0), 0));
assert(isequal(round(2.5), 3));
assert(isequal(round(-2.5), -3));
end

function test_round_complex()
r = round(3.6 + 2.4i);
assert(isequal(real(r), 4));
assert(isequal(imag(r), 2));
end

function test_round_tensor()
x = [1.4 2.5 -0.5 3.6];
assert(isequal(round(x), [1 3 -1 4]));
end

function test_round_n()
% round(x, n) - round to n decimal places
assert(isequal(round(3.14159, 2), 3.14));
assert(isequal(round(3.145, 2), 3.15));
assert(isequal(round(1234, -2), 1200));
end

% ===== Top-level test calls =====

%!jit
test_sin_scalar();
%!jit
test_sin_complex();
%!jit
test_sin_tensor();
%!jit
test_sin_complex_tensor();
%!jit
test_cos_scalar();
%!jit
test_cos_complex();
%!jit
test_cos_tensor();
%!jit
test_cos_complex_tensor();
%!jit
test_tan_scalar();
%!jit
test_tan_complex();
%!jit
test_tan_tensor();
%!jit
test_asin_scalar();
%!jit
test_asin_complex();
%!jit
test_asin_tensor();
%!jit
test_acos_scalar();
%!jit
test_acos_complex();
%!jit
test_acos_tensor();
%!jit
test_atan_scalar();
%!jit
test_atan_complex();
%!jit
test_atan_tensor();
%!jit
test_sinh_scalar();
%!jit
test_sinh_complex();
%!jit
test_sinh_tensor();
%!jit
test_cosh_scalar();
%!jit
test_cosh_complex();
%!jit
test_cosh_tensor();
%!jit
test_tanh_scalar();
%!jit
test_tanh_complex();
%!jit
test_tanh_tensor();
%!jit
test_exp_scalar();
%!jit
test_exp_complex();
%!jit
test_exp_tensor();
%!jit
test_exp_complex_tensor();
%!jit
test_log_scalar();
%!jit
test_log_complex();
%!jit
test_log_tensor();
%!jit
test_log_complex_tensor();
%!jit
test_log2_scalar();
%!jit
test_log2_complex();
%!jit
test_log2_tensor();
%!jit
test_log2_frexp();
%!jit
test_log2_frexp_tensor();
%!jit
test_log10_scalar();
%!jit
test_log10_complex();
%!jit
test_log10_tensor();
%!jit
test_abs_scalar();
%!jit
test_abs_complex();
%!jit
test_abs_tensor();
%!jit
test_abs_complex_tensor();
%!jit
test_sqrt_scalar();
%!jit
test_sqrt_complex();
%!jit
test_sqrt_tensor();
%!jit
test_sqrt_complex_tensor();
%!jit
test_sign_scalar();
%!jit
test_sign_complex();
%!jit
test_sign_tensor();
%!jit
test_sign_complex_tensor();
%!jit
test_floor_scalar();
%!jit
test_floor_complex();
%!jit
test_floor_tensor();
%!jit
test_ceil_scalar();
%!jit
test_ceil_complex();
%!jit
test_ceil_tensor();
%!jit
test_fix_scalar();
%!jit
test_fix_complex();
%!jit
test_fix_tensor();
%!jit
test_round_scalar();
%!jit
test_round_complex();
%!jit
test_round_tensor();
%!jit
test_round_n();

disp('SUCCESS');
