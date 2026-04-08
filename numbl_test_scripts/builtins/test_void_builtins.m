% Builtins that produce no output must not clobber `ans` and must raise
% "Too many output arguments." when the call site expects any outputs.
%
% This mirrors MATLAB: `disp(x)`, `assert(true)`, `fprintf('...')`, etc.
% do not set `ans`, and `r = disp(x)` raises an error.

% ── Preservation of ans ────────────────────────────────────────────────

42;
disp('hi')
assert(ans == 42, 'disp should not overwrite ans');

42;
fprintf('hi\n')
assert(ans == 42, 'fprintf should not overwrite ans');

42;
assert(true)
assert(ans == 42, 'assert should not overwrite ans');

42;
warning('off', 'foo:bar')
assert(ans == 42, 'warning should not overwrite ans');

42;
pause(0)
assert(ans == 42, 'pause should not overwrite ans');

42;
drawnow
assert(ans == 42, 'drawnow should not overwrite ans');

42;
title('t')
assert(ans == 42, 'title should not overwrite ans');

42;
xlabel('x')
assert(ans == 42, 'xlabel should not overwrite ans');

42;
hold on
assert(ans == 42, 'hold should not overwrite ans');

42;
clf
assert(ans == 42, 'clf should not overwrite ans');

% ── fprintf still returns count when assigned ─────────────────────────
n = fprintf('3ch\n');
assert(n == 4, 'fprintf should return byte count when assigned');

% ── Assigning a truly-void builtin errors ─────────────────────────────
err = '';
try; r = disp('x'); catch ME; err = ME.message; end %#ok<NASGU>
assert(strcmp(err, 'Too many output arguments.'), ...
    'r = disp(...) should raise "Too many output arguments."');

err = '';
try; r = assert(true); catch ME; err = ME.message; end %#ok<NASGU>
assert(strcmp(err, 'Too many output arguments.'), ...
    'r = assert(...) should raise "Too many output arguments."');

err = '';
try; r = drawnow; catch ME; err = ME.message; end %#ok<NASGU>
assert(strcmp(err, 'Too many output arguments.'), ...
    'r = drawnow should raise "Too many output arguments."');

err = '';
try; r = hold('on'); catch ME; err = ME.message; end %#ok<NASGU>
assert(strcmp(err, 'Too many output arguments.'), ...
    'r = hold(...) should raise "Too many output arguments."');

err = '';
try; r = grid('on'); catch ME; err = ME.message; end %#ok<NASGU>
assert(strcmp(err, 'Too many output arguments.'), ...
    'r = grid(...) should raise "Too many output arguments."');

disp('SUCCESS');
