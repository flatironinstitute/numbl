% Test function handles

% Store handle in variable
f = @(x) x.^2 + 1;
assert(f(3) == 10);
assert(f(0) == 1);

% Higher-order function
function result = apply_twice(fn, x)
  result = fn(fn(x));
end

double_fn = @(x) x * 2;
assert(apply_twice(double_fn, 3) == 12);

% Closure: capture variable
a = 5;
add_a = @(x) x + a;
assert(add_a(3) == 8);
% Note: MATLAB closures snapshot captured vars; our impl captures by reference (see TODO.md)

% Function handle to builtin
fn_sqrt = @sqrt;
assert(abs(fn_sqrt(9) - 3) < 1e-5);
assert(abs(fn_sqrt(4) - 2) < 1e-5);

disp('SUCCESS')
