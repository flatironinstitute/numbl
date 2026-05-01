% A lambda created inside a function can capture a tensor from the enclosing
% scope. After the function returns, the lambda must still see the captured
% tensor's data — even though the function's local scope has been cleaned up.

f = make_lambda();
v = f(3);
assert(v == 30, sprintf('expected 30, got %g', v));
v = f(1);
assert(v == 10, sprintf('expected 10, got %g', v));
v = f(5);
assert(v == 50, sprintf('expected 50, got %g', v));

% Repeat: after the function call returns the lambda, the buffer the
% lambda reads from must be alive across many invocations.
g = make_lambda();
total = 0;
for k = 1:5
  total = total + g(k);
end
assert(total == 10+20+30+40+50, sprintf('expected 150, got %g', total));

disp('SUCCESS')

function f = make_lambda()
  data = [10, 20, 30, 40, 50];
  f = @(i) data(i);
end
