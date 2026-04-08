% Test that anonymous functions with expression bodies throw
% "Too many output arguments." when more than one output is requested,
% but still allow multi-output propagation when the body is a call to a
% multi-output function.

% --- Expression body: should throw when >1 output is requested ---

% Simple numeric body
f1 = @(t) 42;
try
    [a, b] = f1(0);  %#ok<ASGLU>
    error('testfail:e1', 'expected error for expression body');
catch ME
    assert(contains(ME.message, 'Too many output') || ...
           contains(ME.message, 'Insufficient number of outputs'), ...
        ['unexpected error: ' ME.message]);
end

% Arithmetic expression body (like the chunkie case)
ctr = [0.4; -0.2];
r = 1.3;
fcurve = @(t) ctr + r * [cos(t(:).'); sin(t(:).')];
try
    [x, d, d2] = fcurve(0.0);  %#ok<ASGLU>
    error('testfail:e2', 'expected error for arithmetic body');
catch ME
    assert(contains(ME.message, 'Too many output') || ...
           contains(ME.message, 'Insufficient number of outputs'), ...
        ['unexpected error: ' ME.message]);
end

% Single output still works for expression bodies
y = f1(0);
assert(y == 42, 'single output expression');

xy = fcurve(0.0);
assert(isequal(size(xy), [2 1]), 'fcurve single output size');

% Nested try/catch pattern used by chunkie to probe nargout
fcurve2 = @(t) [cos(t); sin(t)];
try
    [rr, dd, dd2] = fcurve2(0.0);  %#ok<ASGLU>
    nout = 3;
catch
    try
        [rr, dd] = fcurve2(0.0);  %#ok<ASGLU>
        nout = 2;
    catch
        nout = 1;
    end
end
assert(nout == 1, 'nout probe should settle on 1 for expression body');

% --- Call body: multi-output should propagate ---
g = @(x) local_multi_out(x);
[a1, b1] = g(5);
assert(a1 == 10 && b1 == 25, 'call body multi-output');

% Builtin call body (size) can return multiple outputs
h = @(A) size(A);
[nr, nc] = h([1 2 3; 4 5 6]);
assert(nr == 2 && nc == 3, 'size-wrapped anon');

disp('SUCCESS');

function [doubled, squared] = local_multi_out(x)
    doubled = x * 2;
    squared = x^2;
end
