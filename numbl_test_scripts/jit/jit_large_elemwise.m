% Element-wise ops on tensors large enough that the JS-JIT routes them
% through the host's kernels (the interpreter's), and on small ones that
% stay in the JS loops. Both paths must give the same values as a scalar
% loop, including NaN, signed zero, and halfway rounding (transcendentals
% to a relative 1e-15, since libm and V8 may differ in the last bit).

base = [-2.5 -1.5 -0.5 -0 0 0.5 1.5 2.5 NaN Inf -Inf 3.7 -3.7 1e-300];
big = repmat(base, 1, 200);   % 2800 elements
small = base;

for v = {small, big}
    x = v{1}(:);
    [r, s] = ops(x, x(end:-1:1));
    n = numel(x);
    for i = 1:n
        a = x(i);
        b = x(n + 1 - i);
        e = [a + b, a - b, a * b, a / b, a + 2, 3 - a, a * 0.5, 2 / a, ...
             round(a), fix(a), floor(a), ceil(a), sign(a), abs(a), ...
             exp(a), sin(a), cos(a), tanh(a)];
        got = r(i, :);
        bothnan = isnan(got) & isnan(e);
        exact = (got == e) | bothnan;
        assert(all(exact(1:14)), sprintf('exact op, element %d of %d', i, n));
        close = abs(got - e) <= 1e-15 * abs(e) | got == e | bothnan;
        assert(all(close(15:18)), sprintf('transcendental, element %d', i));
        z = got(9:12) == 0;
        assert(isequal(1 ./ got([false(1, 8) z]), 1 ./ e([false(1, 8) z])), ...
            'signed zero');
    end
    assert(isequal(s, [max(x), min(x)]), 'max/min');
end

disp('SUCCESS')

function [r, s] = ops(x, y)
    %!numbl:assert_jit
    r = [x + y, x - y, x .* y, x ./ y, x + 2, 3 - x, x .* 0.5, 2 ./ x, ...
         round(x), fix(x), floor(x), ceil(x), sign(x), abs(x), exp(x), ...
         sin(x), cos(x), tanh(x)];
    s = [max(x), min(x)];
end
