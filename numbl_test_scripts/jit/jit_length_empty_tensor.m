% Regression: C-JIT `length(tensor)` must return 0 for empty tensors that
% carry a non-zero dimension (e.g. zeros(0,5), zeros(5,0), zeros(0,1)).
% The emitter previously returned max(d0, d1) when the tensor had
% hasFreshAlloc, ignoring the emptiness check.

%!numbl:assert_jit
function [l1, l2, l3, l4, l5] = get_lengths()
  a = zeros(0, 5);
  l1 = length(a);
  b = zeros(5, 0);
  l2 = length(b);
  c = zeros(0, 0);
  l3 = length(c);
  d = zeros(0, 1);
  l4 = length(d);
  e = zeros(3, 5);
  l5 = length(e);
end

for k = 1:20
  [l1, l2, l3, l4, l5] = get_lengths();
end

assert(l1 == 0, 'length(zeros(0,5)) must be 0');
assert(l2 == 0, 'length(zeros(5,0)) must be 0');
assert(l3 == 0, 'length(zeros(0,0)) must be 0');
assert(l4 == 0, 'length(zeros(0,1)) must be 0');
assert(l5 == 5, 'length(zeros(3,5)) must be 5');

disp('SUCCESS')
