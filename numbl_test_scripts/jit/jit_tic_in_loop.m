% Bare `tic;` no longer keeps a loop out of the JIT.
%
% numbl's `tic` returned its start time even at nargout 0, so `tic;` bound
% `ans` — which MATLAB does not do, and which made the statement a bare
% non-void expression, something the JIT boundary cannot model. Any loop
% containing a `tic; ... toc` pair therefore ran interpreted, which is
% unfortunate for the loops most likely to contain one.

s = 0;
for k = 1:5
    %!numbl:assert_jit
    tic;
    s = s + k;
    e = toc;
end
assert(s == 15, 'loop body ran');
assert(e >= 0, 'toc returned an elapsed time');

% Command form too.
s2 = 0;
for k = 1:3
    %!numbl:assert_jit
    tic
    s2 = s2 + 1;
end
assert(s2 == 3, 'command-form tic');

% Neither form sets `ans` (MATLAB does not).
clear ans
tic;
assert(~exist('ans', 'var'), 'bare tic must not set ans');

% The assigned form still returns the handle, and toc(handle) works.
t0 = tic;
assert(t0 > 0, 't = tic returns a value');
el = toc(t0);
assert(el >= 0, 'toc(handle)');

% A local named `tic` shadows the builtin.
tic = 7;
assert(tic == 7, 'shadowed tic');

disp('SUCCESS')
