% Test functions with multiple return values

v = [3, 1, 4, 1, 5, 9, 2, 6];
[lo, hi] = minmax(v);
assert(lo == 1);
assert(hi == 9);

[q, r] = divide(17, 5);
assert(q == 3);
assert(r == 2);

% min/max with index output
[val, idx] = min([5, 2, 8, 1, 9]);
assert(val == 1);
assert(idx == 4);

[val, idx] = max([5, 2, 8, 1, 9]);
assert(val == 9);
assert(idx == 5);

disp('SUCCESS')

function [mn, mx] = minmax(v)
  mn = min(v);
  mx = max(v);
end

function [q, r] = divide(a, b)
  q = floor(a / b);
  r = mod(a, b);
end
