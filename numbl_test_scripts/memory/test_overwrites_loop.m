% A tight loop that repeatedly rebinds a tensor variable. Each iteration's
% prior value goes through env.set's release-on-overwrite. The test passes
% if the program runs without corrupting any reads.

x = zeros(1, 100);
for i = 1:1000
  x = (1:100) * i;
end
assert(x(1) == 1000 && x(50) == 50000 && x(100) == 100000, ...
  'final value should reflect the last iteration');

% Same idea, but each iteration's assertion reads the current value to make
% sure the buffer is intact mid-loop (not silently aliased).
acc = 0;
for k = 1:200
  y = ones(1, 50) * k;
  acc = acc + y(1) + y(50);
end
assert(acc == 2 * sum(1:200), sprintf('expected %d, got %g', 2 * sum(1:200), acc));

% Reassign with shape changes. The pool keys by length so each length
% should be handled independently.
for j = 1:50
  z = ones(1, j);
  assert(length(z) == j);
end

% Conditional reassignment — half the iterations reassign, half don't.
w = [1, 2, 3];
for i = 1:100
  if mod(i, 2) == 0
    w = [w, i];
  end
end
assert(length(w) == 3 + 50, 'w should have 3 + 50 elements');

disp('SUCCESS')
