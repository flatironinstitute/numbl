% Compiled code may read a caller's tensor without copying it when the
% function only reads that parameter. The caller's variable and the
% function's results must still behave as copies: writes after the call,
% a parameter returned unchanged, and the same array passed twice.
% Arrays are large enough for the C-JIT to pass them by pointer.

n = 2000;
x = (1:n)';

% Calls from interpreted code (the loop declines: containers.Map), so each
% goes through the call executor.
for k = 1:3
  q = containers.Map();
  s = sum_of(x);
  y = ident(x);
  z = first_plus_last(x, x);
  w = write_first(x, x);
  v = through_struct(x);
  u = calls_ident(x);
  p = write_local(x);
  r = reassign(x);
end
y(1) = -5;
u(2) = -6;
x(3) = -7;
check(x, s, y, z, w, v, u, p, r, n);

% The same calls from a compiled loop (internal call sites), with `x` an
% input of the loop.
x = (1:n)';
for k = 1:3
  s = sum_of(x);
  y = ident(x);
  z = first_plus_last(x, x);
  w = write_first(x, x);
  v = through_struct(x);
  u = calls_ident(x);
  p = write_local(x);
  r = reassign(x);
  y(4) = -4;
  assert(x(4) == 4);
end
y(4) = 4;
y(1) = -5;
u(2) = -6;
x(3) = -7;
check(x, s, y, z, w, v, u, p, r, n);

% A loop that only reads its input, then the caller writes to it.
x = (1:n)';
t = 0;
for k = 1:n
  t = t + x(k);
end
x(1) = -1;
assert(t == n * (n + 1) / 2);
assert(x(1) == -1 && x(2) == 2);

% An element-wise statement whose output is also an input.
a = (1:70000)';
b = 2 * a;
c = a;
for k = 1:4
  c = c .* a + b;
end
assert(a(1) == 1 && a(end) == 70000 && b(end) == 140000);
c(1) = 0;
assert(a(1) == 1);

% An error raised in a function that borrowed its argument.
x = (1:n)';
for k = 1:3
  q = containers.Map();
  try
    r = index_past_end(x);
    assert(false);
  catch
  end
end
assert(isequal(x, (1:n)'));

disp('SUCCESS')

function check(x, s, y, z, w, v, u, p, r, n)
  assert(s == n * (n + 1) / 2);
  assert(z == 1 + n);
  assert(w(1) == 0 && w(2) == n + 1);
  assert(v(1) == 0 && v(2) == 2);
  assert(p == 2);
  assert(r == 2);
  assert(x(1) == 1 && x(2) == 2 && x(3) == -7 && x(4) == 4);
  assert(y(1) == -5 && y(2) == 2 && y(3) == 3);
  assert(u(1) == -9 && u(2) == -6 && u(3) == 3);
end

function s = sum_of(a)
  s = sum(a);
end

function a = ident(a)
end

function r = first_plus_last(a, b)
  r = a(1) + b(end);
end

function a = write_first(a, b)
  a(1) = 0;
  a(2) = b(1) + b(end);
end

function r = write_local(a)
  a(1) = 0;
  r = a(1) + a(2);
end

function r = reassign(a)
  a = a * 2;
  r = a(1);
end

function r = through_struct(a)
  st = struct('f', a);
  st.f(1) = 0;
  r = st.f;
end

function r = calls_ident(a)
  r = ident(a);
  r(1) = -9;
  assert(a(1) ~= -9);
end

function r = index_past_end(a)
  r = a(numel(a) + 1);
end
