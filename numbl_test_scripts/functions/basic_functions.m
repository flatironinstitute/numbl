% Basic user-defined functions

assert(square(3) == 9)
assert(square(5) == 25)

assert(add(3, 4) == 7)
assert(add(-1, 1) == 0)

assert(abs(factorial(5) - 120) < 0.5)

% Function with multiple return values
[mn, mx] = minmax([3, 1, 4, 1, 5, 9, 2]);
assert(mn == 1)
assert(mx == 9)

disp('SUCCESS')

function y = square(x)
  y = x * x;
end

function z = add(a, b)
  z = a + b;
end

function [lo, hi] = minmax(v)
  lo = min(v);
  hi = max(v);
end
