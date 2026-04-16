% MATLAB passes function arguments by value (copy-on-write). When a
% function mutates a passed argument, the caller's copy is unaffected.
%
% Each case calls the mutating helper multiple times with the same
% caller-side variable. A single call happens to work, because the tensor
% wrapper is already marked shared (refcount > 1). But on the second call
% the prior COW decremented the wrapper's refcount to 1, so the mutation
% lands directly on the caller's data.

% ── Scalar index assignment on a tensor argument ───────────────────────
v = [1, 2, 3, 4, 5];
y = mutate_scalar(v);
y = mutate_scalar(v);
y = mutate_scalar(v);
assert(isequal(v, [1, 2, 3, 4, 5]), 'v should be unchanged after mutate_scalar');
assert(isequal(y, [1, 2, 99, 4, 5]), 'returned y should have the mutation');

% ── 2D index assignment on a matrix argument ───────────────────────────
M = [1, 2; 3, 4];
y = mutate_2d(M);
y = mutate_2d(M);
y = mutate_2d(M);
assert(isequal(M, [1, 2; 3, 4]), 'M should be unchanged after mutate_2d');
assert(isequal(y, [1, 77; 3, 4]), 'returned y should have M(1,2)=77');

% ── Slice assignment ───────────────────────────────────────────────────
v = [1, 2, 3, 4, 5];
y = mutate_slice(v);
y = mutate_slice(v);
y = mutate_slice(v);
assert(isequal(v, [1, 2, 3, 4, 5]), 'v should be unchanged after mutate_slice');
assert(isequal(y, [1, 2, 88, 88, 5]), 'returned y should have slice mutation');

% ── Growing a tensor argument ──────────────────────────────────────────
v = [1, 2, 3];
y = mutate_grow(v);
y = mutate_grow(v);
y = mutate_grow(v);
assert(isequal(v, [1, 2, 3]), 'v should be unchanged after mutate_grow');
assert(length(y) == 5, 'returned y should have grown to length 5');

% ── Struct field mutation ──────────────────────────────────────────────
s.x = 10;
s.y = 20;
t = mutate_struct(s);
t = mutate_struct(s);
t = mutate_struct(s);
assert(s.x == 10, 's.x should be unchanged after mutate_struct');
assert(t.x == 55, 'returned t.x should have the mutation');

% ── Cell array mutation ────────────────────────────────────────────────
c = {1, 2, 3};
d = mutate_cell(c);
d = mutate_cell(c);
d = mutate_cell(c);
assert(isequal(c{2}, 2), 'c{2} should be unchanged after mutate_cell');
assert(isequal(d{2}, 222), 'returned d{2} should have the mutation');

disp('SUCCESS')

function out = mutate_scalar(v)
  v(3) = 99;
  out = v;
end

function out = mutate_2d(M)
  M(1, 2) = 77;
  out = M;
end

function out = mutate_slice(v)
  v(3:4) = 88;
  out = v;
end

function out = mutate_grow(v)
  v(5) = 0;
  out = v;
end

function out = mutate_struct(s)
  s.x = 55;
  out = s;
end

function out = mutate_cell(c)
  c{2} = 222;
  out = c;
end
