% Large element-wise statements that run on the interpreter (here in a
% loop the JIT declines, because of num2str) are compiled on their own
% once hot, as fused kernels. They must assign exactly what the same
% computation gives one operation at a time, and raise where it raises.

n = 70000;
a = linspace(-3, 3, n)';
b = cos(a);
c = a .^ 2 - 1;
c(5) = NaN;
c(7) = -0;
for k = 1:6
    lbl = num2str(k);
    y = a .* b + c .* k - a;
    s = sum(a .* b - k .* b);
    m = mean(abs(a) .* 0.5 + k);
    r1 = a .* b;
    r2 = c .* k;
    r3 = r1 + r2;
    yref = r3 - a;
    assert(eqn(y, yref), ['statement ' lbl]);
    u = a .* b;
    u = u - k .* b;
    assert(eqn(s, sum(u)), ['sum ' lbl]);
    v = abs(a);
    v = v .* 0.5;
    v = v + k;
    assert(isequal(m, mean(v)), ['mean ' lbl]);
end

% A variable named like a function makes the "call" an index.
abs = 2 * ones(n, 1);
for k = 1:4
    lbl = num2str(k);
    z = abs(3) + a .* b;
    assert(isequal(z, 2 + a .* b), ['shadowed call ' lbl]);
end
clear abs

% Shapes that stop agreeing after the statement compiled.
x = ones(n, 1);
for k = 1:5
    lbl = num2str(k);
    if k < 4
        w = ones(n, 1);
    else
        w = ones(n + 1, 1);
    end
    failed = false;
    try
        q = x .* 2 + w .* 3;
    catch
        failed = true;
    end
    assert(failed == (k >= 4), ['mismatch ' lbl]);
    if ~failed
        assert(isequal(q, 5 * ones(n, 1)), ['value ' lbl]);
    end
end

disp('SUCCESS')

function t = eqn(p, q)
    % Equal values, with NaN matching NaN.
    t = isequal(size(p), size(q)) && isequal(isnan(p), isnan(q)) && ...
        isequal(p(~isnan(p)), q(~isnan(q)));
end
