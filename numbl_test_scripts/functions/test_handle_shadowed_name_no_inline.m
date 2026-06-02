% Regression: a function handle that references a name which is a FUNCTION
% (or undefined) at handle-creation time, but later is ALSO a loop variable,
% must keep its definition-time meaning when the enclosing loop is JIT'd.
%
% The capture-free-handle inliner relocates a handle's `@...` definition
% into the loop's synthetic scope. It tested captures only against the
% handle's def env, so a name that was a function/undefined at definition
% (not an env variable) looked "capture-free" — but the loop scope binds it
% as a variable (a loop input), so re-lowering re-resolved it to the loop
% variable. That silently turned the function call `sq(t)` into an array
% index, or masked an undefined-variable error. The inliner now declines on
% such a collision, so the loop falls back to the interpreter and keeps
% MATLAB semantics on every --opt level.

% (1) handle CALLS a function `sq` that is later shadowed by a vector var.
g = @(t) sq(t) + 1;
sq = [10 20 30 40 50];
acc = 0;
for i = 1:3
    acc = acc + g(i) + sq(i);     % g(i) calls sq()=i^2; +1; plus sq(i) vector
end
% g(i) = i^2 + 1 -> 2,5,10 (sum 17); sq(i) -> 10+20+30 (sum 60); total 77
assert(acc == 77, 'handle function-call must not become array index');

% (2) handle references a 0-arg function `two`, later shadowed by a var.
h = @(t) t + two;
two = 1000;
acc2 = 0;
for i = 1:3
    acc2 = acc2 + h(i) + two;     % h(i) = i + two()=2; plus two=1000 var
end
% h(i) = i + 2 -> 3,4,5; plus 1000 each -> 1003+1004+1005 = 3012
assert(acc2 == 3012, 'handle 0-arg-function ref must not capture loop var');

disp('SUCCESS')

function out = sq(x)
    out = x .^ 2;
end

function v = two()
    v = 2;
end
