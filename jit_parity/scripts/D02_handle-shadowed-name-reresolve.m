% TEST: a function handle `g = @(t) sq(t) + 1` defined while `sq` is a
% FUNCTION, then `sq` is reassigned to a vector and read inside a JIT'd loop.
% MATLAB / opt0 (interp): g(i) keeps calling the function sq -> 77.
% opt1 (JS-JIT, before fix): 123   <-- DIVERGED
% opt2 (C-JIT, before fix):  123   <-- DIVERGED
% DIVERGING MODE: opt1 and opt2 (silent wrong value).
%
% Cause: the capture-free-handle inliner (handleInline.ts) relocated the
%   handle into the loop's synthetic scope and tested captures only against
%   the handle's def env. `sq` was a function at definition (not an env
%   variable) so it looked capture-free, but the loop scope binds `sq` as a
%   variable (loop input), so re-lowering turned the call sq(t) into an
%   index into the captured vector. FIX: the inliner declines when a free
%   body name collides with a loop-scope name, so the loop falls back to the
%   interpreter and all opts agree.
g = @(t) sq(t) + 1;
sq = [10 20 30 40 50];
acc = 0;
for i = 1:3
    acc = acc + g(i) + sq(i);
end
disp(acc);

function out = sq(x)
    out = x .^ 2;
end
