% Two-argument max and min with a tensor argument: tensor-scalar,
% scalar-tensor, tensor-tensor and broadcast forms, NaN handling (the
% non-NaN operand wins), at top level and through a JIT-compiled function.

x = [-2 -1 0 1 2];
assert(isequal(max(x, 0), [0 0 0 1 2]));
assert(isequal(min(x, 0), [-2 -1 0 0 0]));
assert(isequal(max(0, x), [0 0 0 1 2]));
assert(isequal(min(1, x), [-2 -1 0 1 1]));
assert(isequal(max(x, fliplr(x)), [2 1 0 1 2]));
assert(isequal(min(x, fliplr(x)), [-2 -1 0 -1 -2]));

% broadcast: a row against a column
B = max([1 2 3], [0; 2]);
assert(isequal(B, [1 2 3; 2 2 3]));
C = min([1 2 3], [0; 2]);
assert(isequal(C, [0 0 0; 1 2 2]));

% NaN is treated as missing
v = max([NaN 1 NaN], [2 NaN NaN]);
assert(v(1) == 2 && v(2) == 1 && isnan(v(3)));
w = min([NaN 5], 3);
assert(isequal(w, [3 3]));

% through a function, called in a loop, so the JIT compiles it
s = 0;
for k = 1:200
  y = me_ramp(sin(k * (1:8)));
  s = s + sum(y);
end
ref = 0;
for k = 1:200
  t = sin(k * (1:8));
  ref = ref + sum(2 * (t .* (t > 0)) - 1);
end
assert(abs(s - ref) < 1e-10);

% the one-argument and dimension forms are still reductions
assert(max([3 1 2]) == 3);
assert(isequal(min([3 1; 2 4], [], 2), [1; 2]));

disp('SUCCESS');

function y = me_ramp(x)
  y = 2 * max(x, 0) - 1;
end
