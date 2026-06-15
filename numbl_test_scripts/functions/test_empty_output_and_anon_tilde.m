% Parser regression:
%   - `function [] = name(...)` explicit empty output list (no outputs)
%   - `~` placeholder parameters in anonymous functions

% Empty-output-list function runs and produces no output.
42;
empty_out(5);
assert(ans == 42, 'empty-output function should not set ans');

% Anonymous function with ~ placeholders ignores those arguments.
f = @(~, ~) 42;
assert(f(1, 2) == 42, 'anon with all-tilde params');

g = @(~, x) x * 2;
assert(g(99, 5) == 10, 'anon with leading tilde param');

disp('SUCCESS');

function [] = empty_out(x)
  assert(x == 5, 'argument passed through');
end
