% R: error(id, fmt, ...) raised inside a JIT'd loop in a function,
% uncaught: same message and location at every --opt level.
disp(checked_sum(3));
disp(checked_sum(10));

function s = checked_sum(n)
  s = 0;
  for k = 1:n
    if k > 5
      error('parity:tooBig', 'k=%d exceeds the limit of %d', k, 5);
    end
    s = s + k;
  end
end
