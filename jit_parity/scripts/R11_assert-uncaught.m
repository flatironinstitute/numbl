% R: a failing formatted assert inside a JIT'd loop, uncaught. MATLAB's
% assert(cond, id, fmt, args...) formats the message and sets the id.
x = 0;
for k = 1:6
  x = x + k;
  assert(x < 10, 'parity:limit', 'x=%d reached the limit at k=%d', x, k);
end
disp(x)
