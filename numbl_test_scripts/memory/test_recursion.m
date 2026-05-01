% Recursive functions create a chain of frames each holding refcount-bumped
% aliases of the same buffer. Each return must release exactly the bump it
% added, leaving the caller's view intact.

v = (1:50)';
out = recursive_sum(v, 50);
assert(out == sum(1:50), sprintf('expected %d, got %g', sum(1:50), out));
assert(isequal(v, (1:50)'), 'caller v must be unchanged after recursion');

% Recursion that mutates at each level — COW must trigger every time.
m = [1, 2, 3, 4, 5];
out2 = recursive_mutate(m, 1, 5);
assert(isequal(m, [1, 2, 3, 4, 5]), 'caller m must be unchanged after recursive mutate');
% Each of 5 frames adds 100 to v(frame), so out2(i) = m(i) + 100.
assert(isequal(out2, [101, 102, 103, 104, 105]), ...
  sprintf('expected [101..105], got [%s]', strjoin(arrayfun(@num2str, out2, 'UniformOutput', false), ',')));

% Recursion building up a tensor and returning it.
tensor_out = build(10);
assert(isequal(tensor_out, 1:10), 'recursive build should produce 1..10');

disp('SUCCESS')

function s = recursive_sum(v, n)
  if n == 0
    s = 0;
    return;
  end
  s = v(n) + recursive_sum(v, n - 1);
end

function r = recursive_mutate(v, depth, maxDepth)
  v(depth) = v(depth) + 100;
  if depth >= maxDepth
    r = v;
  else
    r = recursive_mutate(v, depth + 1, maxDepth);
  end
end

function t = build(n)
  if n == 0
    t = [];
  else
    t = [build(n-1), n];
  end
end
