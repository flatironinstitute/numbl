% ===== zeros =====

function test_zeros_scalar()
assert(isequal(zeros(), 0));
assert(isequal(zeros(1), 0));
end

function test_zeros_matrix()
z = zeros(2, 3);
assert(isequal(size(z), [2, 3]));
assert(isequal(z(1,1), 0));
assert(isequal(z(2,3), 0));
end

function test_zeros_square()
z = zeros(3);
assert(isequal(size(z), [3, 3]));
assert(isequal(z(1,1), 0));
end

function test_zeros_vector_arg()
z = zeros([2, 4]);
assert(isequal(size(z), [2, 4]));
assert(isequal(z(1,1), 0));
end

% ===== ones =====

function test_ones_scalar()
assert(isequal(ones(), 1));
assert(isequal(ones(1), 1));
end

function test_ones_matrix()
o = ones(2, 3);
assert(isequal(size(o), [2, 3]));
assert(isequal(o(1,1), 1));
assert(isequal(o(2,3), 1));
end

function test_ones_square()
o = ones(3);
assert(isequal(size(o), [3, 3]));
assert(isequal(o(2,2), 1));
end

% ===== eye =====

function test_eye_default()
e = eye();
assert(isequal(size(e), [1, 1]));
assert(isequal(e(1,1), 1));
end

function test_eye_square()
e = eye(3);
assert(isequal(size(e), [3, 3]));
assert(isequal(e(1,1), 1));
assert(isequal(e(1,2), 0));
assert(isequal(e(2,2), 1));
assert(isequal(e(3,3), 1));
end

function test_eye_rectangular()
e = eye(2, 3);
assert(isequal(size(e), [2, 3]));
assert(isequal(e(1,1), 1));
assert(isequal(e(2,2), 1));
assert(isequal(e(1,3), 0));
end

% ===== linspace =====

function test_linspace_basic()
x = linspace(0, 1, 5);
assert(isequal(size(x), [1, 5]));
assert(abs(x(1) - 0) < 1e-15);
assert(abs(x(5) - 1) < 1e-15);
assert(abs(x(3) - 0.5) < 1e-15);
end

function test_linspace_two_args()
x = linspace(0, 10);
assert(isequal(size(x), [1, 100]));
assert(abs(x(1) - 0) < 1e-15);
assert(abs(x(100) - 10) < 1e-15);
end

function test_linspace_single_point()
x = linspace(5, 5, 1);
assert(isequal(size(x), [1, 1]));
assert(abs(x(1) - 5) < 1e-15);
end

% ===== logspace =====

function test_logspace_basic()
x = logspace(1, 3, 3);
assert(isequal(size(x), [1, 3]));
assert(abs(x(1) - 10) < 1e-10);
assert(abs(x(2) - 100) < 1e-10);
assert(abs(x(3) - 1000) < 1e-10);
end

function test_logspace_two_args()
x = logspace(0, 2);
assert(isequal(size(x), [1, 50]));
assert(abs(x(1) - 1) < 1e-10);
assert(abs(x(50) - 100) < 1e-10);
end

% ===== Top-level test calls =====

%!jit
test_zeros_scalar();
%!jit
test_zeros_matrix();
%!jit
test_zeros_square();
%!jit
test_zeros_vector_arg();

%!jit
test_ones_scalar();
%!jit
test_ones_matrix();
%!jit
test_ones_square();

%!jit
test_eye_default();
%!jit
test_eye_square();
%!jit
test_eye_rectangular();

%!jit
test_linspace_basic();
%!jit
test_linspace_two_args();
%!jit
test_linspace_single_point();

%!jit
test_logspace_basic();
%!jit
test_logspace_two_args();

disp('SUCCESS');
