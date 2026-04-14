% Variant of tensor_ops_benchmark.m with small N and many more trials.
% At small N the per-op buffer allocation / wrapper creation dominates
% over the inner elementwise compute, so this benchmark is the sensitive
% one for the JIT in-place reuse optimization.
%
% Run:
%   npx tsx src/cli.ts run examples/tensor_ops_benchmark_small.m
%   matlab -batch "run('examples/tensor_ops_benchmark_small.m')"

N = 1000;
trials = 200000;

fprintf('N=%d, trials=%d\n', N, trials);
fprintf('----------------------------------------\n');

t = linspace(-1.0, 1.0, N)';
x = sin(3.1 .* t) .* 0.9;
y = cos(2.7 .* t + 0.4) .* 0.8;
zr = sin(5.0 .* t);
zi = cos(4.2 .* t);
z = zr + 1i * zi;

% ── 1. Real binary element-wise ───────────────────────────────────────
tic;
for k = 1:trials
    r = x + y;
    r = r - 0.5 .* x;
    r = r .* y + 3.0;
    r = r ./ (1 + abs(y));
end
t1 = toc;

% ── 2. Real unary element-wise ────────────────────────────────────────
tic;
for k = 1:trials
    u = exp(-x .* x);
    u = u .* cos(5 .* x);
    u = u + sin(x + 1) .* sin(x + 1);
    u = sqrt(abs(u));
    u = log(1 + u);
    u = tanh(u);
end
t2 = toc;

% ── 3. Comparisons ────────────────────────────────────────────────────
tic;
count = 0;
for k = 1:trials
    c1 = double(x > 0);
    c2 = double(y < 0.5);
    count = count + sum(c1 .* c2);
    count = count + sum(double(x == y));
    count = count + sum(double(x ~= y));
    count = count + sum(double(x <= 0.3));
    count = count + sum(double(x >= -0.3));
end
t3 = toc;

% ── 4. Complex binary + unary ─────────────────────────────────────────
tic;
for k = 1:trials
    w = z + 1;
    w = w .* z;
    w = w ./ (abs(z) + 1);
    w = exp(w);
    w = sqrt(w);
end
t4 = toc;

total = t1 + t2 + t3 + t4;

fprintf('Real binary elemwise:   %7.3f s\n', t1);
fprintf('Real unary elemwise:    %7.3f s\n', t2);
fprintf('Comparisons:            %7.3f s\n', t3);
fprintf('Complex elemwise:       %7.3f s\n', t4);
fprintf('----------------------------------------\n');
fprintf('Total:                  %7.3f s\n', total);

fprintf('\nCheck values (should match between numbl and MATLAB):\n');
fprintf('  sum(r)       = %.10g\n', sum(r));
fprintf('  sum(u)       = %.10g\n', sum(u));
fprintf('  count        = %d\n', count);
fprintf('  sum(real(w)) = %.10g\n', sum(real(w)));
fprintf('  sum(imag(w)) = %.10g\n', sum(imag(w)));
disp('SUCCESS');
