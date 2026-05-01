% Persistent variable holding a tensor: must survive across many calls,
% support in-place mutation (MATLAB semantics), and allow whole-tensor
% reassignment. The buffer the persistent points at must never be
% prematurely pooled.

% Case 1: in-place index mutation across many calls.
for i = 1:50
  v = accum(i);
end
assert(length(v) == 50, sprintf('expected length 50, got %d', length(v)));
for i = 1:50
  assert(v(i) == i, sprintf('expected v(%d)=%d, got %g', i, i, v(i)));
end

% Case 2: whole-tensor reassignment of a persistent.
for k = 1:20
  out = grow(k);
end
assert(length(out) == 20, 'persistent should hold the latest grown vector');
for j = 1:20
  assert(out(j) == j*j, sprintf('expected out(%d)=%d, got %g', j, j*j, out(j)));
end

% Case 3: persistent that is read but never written.
val = peek_default();
assert(isempty(val), 'first peek should return the default empty');
val2 = peek_default();
assert(isempty(val2), 'second peek should also return empty');

% Case 4: persistent shared via COW with a local copy. Mutating the local
% must not affect the persistent.
[got_p, got_local] = persistent_cow_split();
assert(got_p(2) == 2, 'persistent value should be unchanged');
assert(got_local(2) == 99, 'local copy should reflect the mutation');

disp('SUCCESS')

function out = accum(i)
  persistent v
  v(i) = i;
  out = v;
end

function out = grow(k)
  persistent stash
  stash = (1:k).^2;
  out = stash;
end

function out = peek_default()
  persistent untouched
  out = untouched;
end

function [p, local] = persistent_cow_split()
  persistent shared
  if isempty(shared)
    shared = [1, 2, 3, 4, 5];
  end
  local = shared;
  local(2) = 99;
  p = shared;
end
