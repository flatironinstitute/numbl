% TEST: element-wise kernels with a fused full reduction (sum / prod /
% mean of an element-wise tree, accumulated in linear order like the
% reduction helpers), and the statement shape: large element-wise
% assignments in a loop the JIT declines (num2str) are compiled on their
% own once hot. Values must match the interpreter bit for bit, NaN and
% signed zero included.
n = 70000;
a = linspace(-3, 3, n)';
b = cos(3 * a);
c = a .* a - 1;
c(11) = NaN;
c(12) = -0;
for k = 1:5
    lbl = num2str(k);
    y = a .* b + c .* k - a;
    s = sum(a .* b - k .* b);
    p = prod(abs(b) .* 0.5 + 0.75);
    m = mean(abs(a) .* 0.5 + k);
    fprintf('%s %.17g %.17g %.17g %.17g %.17g\n', lbl, s, p, m, y(10), y(n));
    fprintf('%.17g %.17g %.17g\n', y(11), 1 / y(12), sum(y(13:end)));
end
x = linspace(0, 1, 9);
for k = 1:3
    t = sum(x .* k - 0.5);
    u = mean(-x .* 2 + k);
    fprintf('%.17g %.17g\n', t, u);
end
