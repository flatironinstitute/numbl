% ===== atan2 =====

function test_atan2_scalar()
assert(isequal(atan2(0, 1), 0));
assert(isequal(atan2(1, 0), pi/2));
assert(isequal(atan2(0, -1), pi));
assert(isequal(atan2(-1, 0), -pi/2));
assert(isequal(atan2(1, 1), pi/4));
assert(isequal(atan2(-1, -1), -3*pi/4));
end

function test_atan2_tensor()
assert(isequal(atan2([1 0 -1], [0 1 0]), [pi/2 0 -pi/2]));
assert(isequal(atan2([1; -1], [1; -1]), [pi/4; -3*pi/4]));
assert(isequal(atan2([1 0; -1 0], [0 1; 0 -1]), [pi/2 0; -pi/2 pi]));
% scalar + tensor broadcast
assert(isequal(atan2(1, [1 0 -1]), [pi/4 pi/2 pi - atan2(1, 1)]));
assert(isequal(atan2([1 -1 0], 1), [pi/4 -pi/4 0]));
end

% ===== min =====

function test_min_scalar()
assert(isequal(min(3, 5), 3));
assert(isequal(min(-2, 1), -2));
assert(isequal(min(0, 0), 0));
end

function test_min_reduction()
assert(isequal(min([3 5 2]), 2));
assert(isequal(min([7 1 4 3]), 1));
assert(isequal(min([3 5; 1 4]), [1 4]));
assert(isequal(min([9 2 6; 3 8 1]), [3 2 1]));
end

function test_min_dim()
assert(isequal(min([3 5; 1 4], [], 2), [3; 1]));
assert(isequal(min([9 2 6; 3 8 1], [], 2), [2; 1]));
end

function test_min_elemwise()
assert(isequal(min([3 5 7], [4 2 6]), [3 2 6]));
assert(isequal(min([1; 2; 3], [3; 1; 2]), [1; 1; 2]));
assert(isequal(min([3 5 7], 4), [3 4 4]));
assert(isequal(min(4, [3 5 7]), [3 4 4]));
end

function test_min_complex()
assert(isequal(min(1+2i, 3), 1+2i));
assert(isequal(min(3+4i, 2), 2));
assert(isequal(min([3+4i 1+1i 2]), 1+1i));
end

% ===== max =====

function test_max_scalar()
assert(isequal(max(3, 5), 5));
assert(isequal(max(-2, 1), 1));
assert(isequal(max(0, 0), 0));
end

function test_max_reduction()
assert(isequal(max([3 5 2]), 5));
assert(isequal(max([7 1 4 3]), 7));
assert(isequal(max([3 5; 1 4]), [3 5]));
assert(isequal(max([9 2 6; 3 8 1]), [9 8 6]));
end

function test_max_dim()
assert(isequal(max([3 5; 1 4], [], 2), [5; 4]));
assert(isequal(max([9 2 6; 3 8 1], [], 2), [9; 8]));
end

function test_max_elemwise()
assert(isequal(max([3 5 7], [4 2 6]), [4 5 7]));
assert(isequal(max([1; 2; 3], [3; 1; 2]), [3; 2; 3]));
assert(isequal(max([3 5 7], 4), [4 5 7]));
assert(isequal(max(4, [3 5 7]), [4 5 7]));
end

function test_max_complex()
assert(isequal(max(1+2i, 3), 3));
assert(isequal(max(3+4i, 2), 3+4i));
assert(isequal(max([3+4i 1+1i 2]), 3+4i));
end

% ===== mod =====

function test_mod_scalar()
assert(isequal(mod(7, 3), 1));
assert(isequal(mod(10, 5), 0));
assert(isequal(mod(-1, 3), 2));
assert(isequal(mod(7, -3), -2));
assert(isequal(mod(-7, -3), -1));
assert(isequal(mod(0, 5), 0));
end

function test_mod_tensor()
assert(isequal(mod([7 10 -1], 3), [1 1 2]));
assert(isequal(mod([7; 10], [3; 4]), [1; 2]));
assert(isequal(mod(10, [3 4 5]), [1 2 0]));
assert(isequal(mod([10 20 30], 7), [3 6 2]));
end

% ===== rem =====

function test_rem_scalar()
assert(isequal(rem(7, 3), 1));
assert(isequal(rem(10, 5), 0));
assert(isequal(rem(-1, 3), -1));
assert(isequal(rem(7, -3), 1));
assert(isequal(rem(-7, -3), -1));
assert(isequal(rem(0, 5), 0));
end

function test_rem_tensor()
assert(isequal(rem([7 10 -1], 3), [1 1 -1]));
assert(isequal(rem([7; 10], [3; 4]), [1; 2]));
assert(isequal(rem(10, [3 4 5]), [1 2 0]));
assert(isequal(rem([10 20 30], 7), [3 6 2]));
end

% ===== power =====

function test_power_scalar()
assert(isequal(power(2, 3), 8));
assert(isequal(power(3, 2), 9));
assert(isequal(power(5, 0), 1));
assert(isequal(power(4, 0.5), 2));
assert(abs(power(2, -1) - 0.5) < 1e-15);
end

function test_power_tensor()
assert(isequal(power([2 3 4], 2), [4 9 16]));
assert(isequal(power(2, [1 2 3]), [2 4 8]));
assert(isequal(power([2 3], [3 2]), [8 9]));
assert(isequal(power([1 2; 3 4], 2), [1 4; 9 16]));
end

%!jit
test_atan2_scalar();
%!jit
test_atan2_tensor();
%!jit
test_min_scalar();
%!jit
test_min_reduction();
%!jit
test_min_dim();
%!jit
test_min_elemwise();
%!jit
test_min_complex();
%!jit
test_max_scalar();
%!jit
test_max_reduction();
%!jit
test_max_dim();
%!jit
test_max_elemwise();
%!jit
test_max_complex();
%!jit
test_mod_scalar();
%!jit
test_mod_tensor();
%!jit
test_rem_scalar();
%!jit
test_rem_tensor();
%!jit
test_power_scalar();
%!jit
test_power_tensor();

disp('SUCCESS');
