% ===== isnan =====

function test_isnan_scalar()
assert(isequal(isnan(NaN), 1));
assert(isequal(isnan(0), 0));
assert(isequal(isnan(1), 0));
assert(isequal(isnan(Inf), 0));
assert(isequal(isnan(-Inf), 0));
end

function test_isnan_complex()
assert(isequal(isnan(NaN + 1i), 1));
assert(isequal(isnan(1 + NaN*1i), 1));
assert(isequal(isnan(NaN + NaN*1i), 1));
assert(isequal(isnan(1 + 2i), 0));
end

function test_isnan_tensor()
x = [1 NaN 3 NaN];
r = isnan(x);
assert(isequal(r, [0 1 0 1]));
% 2D
x2 = [1 NaN; NaN 4];
r2 = isnan(x2);
assert(isequal(r2, [0 1; 1 0]));
end

function test_isnan_complex_tensor()
z = [1+2i NaN+1i 3+0i];
r = isnan(z);
assert(isequal(r, [0 1 0]));
end

% ===== isinf =====

function test_isinf_scalar()
assert(isequal(isinf(Inf), 1));
assert(isequal(isinf(-Inf), 1));
assert(isequal(isinf(0), 0));
assert(isequal(isinf(1), 0));
assert(isequal(isinf(NaN), 0));
end

function test_isinf_complex()
assert(isequal(isinf(Inf + 1i), 1));
assert(isequal(isinf(1 + Inf*1i), 1));
assert(isequal(isinf(1 + 2i), 0));
end

function test_isinf_tensor()
x = [1 Inf -Inf 0 NaN];
r = isinf(x);
assert(isequal(r, [0 1 1 0 0]));
% 2D
x2 = [Inf 0; 1 -Inf];
r2 = isinf(x2);
assert(isequal(r2, [1 0; 0 1]));
end

function test_isinf_complex_tensor()
z = [1+2i Inf+0i 0+Inf*1i];
r = isinf(z);
assert(isequal(r, [0 1 1]));
end

% ===== isfinite =====

function test_isfinite_scalar()
assert(isequal(isfinite(1), 1));
assert(isequal(isfinite(0), 1));
assert(isequal(isfinite(-3.5), 1));
assert(isequal(isfinite(Inf), 0));
assert(isequal(isfinite(-Inf), 0));
assert(isequal(isfinite(NaN), 0));
end

function test_isfinite_complex()
assert(isequal(isfinite(1 + 2i), 1));
assert(isequal(isfinite(Inf + 1i), 0));
assert(isequal(isfinite(1 + Inf*1i), 0));
end

function test_isfinite_tensor()
x = [1 Inf NaN 0 -Inf];
r = isfinite(x);
assert(isequal(r, [1 0 0 1 0]));
% 2D
x2 = [1 Inf; NaN 2];
r2 = isfinite(x2);
assert(isequal(r2, [1 0; 0 1]));
end

function test_isfinite_complex_tensor()
z = [1+2i Inf+0i 3+0i];
r = isfinite(z);
assert(isequal(r, [1 0 1]));
end

% ===== isreal =====

function test_isreal_scalar()
assert(isequal(isreal(1), 1));
assert(isequal(isreal(0), 1));
assert(isequal(isreal(-3.5), 1));
end

function test_isreal_complex()
assert(isequal(isreal(1 + 2i), 0));
assert(isequal(isreal(3i), 0));
end

function test_isreal_tensor()
assert(isequal(isreal([1 2 3]), 1));
end

function test_isreal_complex_tensor()
assert(isequal(isreal([1+2i 3]), 0));
end

% ===== predicate results in arithmetic =====

function test_predicate_arithmetic()
% predicates return logical values that should work in arithmetic
x = isnan(NaN) + 1;
assert(isequal(x, 2));
x = isfinite(1) * 5;
assert(isequal(x, 5));
end

% ===== predicate results in comparisons =====

function test_predicate_comparison()
% predicates should compare correctly with numbers
assert(isnan(NaN) == 1);
assert(isnan(0) == 0);
assert(isinf(Inf) == 1);
assert(isfinite(1) == 1);
assert(isfinite(Inf) == 0);
end

% ===== Top-level test calls =====

%!jit
test_isnan_scalar();
%!jit
test_isnan_complex();
%!jit
test_isnan_tensor();
%!jit
test_isnan_complex_tensor();
%!jit
test_isinf_scalar();
%!jit
test_isinf_complex();
%!jit
test_isinf_tensor();
%!jit
test_isinf_complex_tensor();
%!jit
test_isfinite_scalar();
%!jit
test_isfinite_complex();
%!jit
test_isfinite_tensor();
%!jit
test_isfinite_complex_tensor();
%!jit
test_isreal_scalar();
%!jit
test_isreal_complex();
%!jit
test_isreal_tensor();
%!jit
test_isreal_complex_tensor();
%!jit
test_predicate_arithmetic();
%!jit
test_predicate_comparison();

disp('SUCCESS');
