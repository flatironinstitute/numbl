function tensor_ops_bench()
    % Tensor element-wise + reduction benchmark: numbl --opt 0/1/2 vs MATLAB
    % vs Octave. Companion to scalar_bench.m.
    %
    % Each "kernel" function is small enough that the C-JIT can compile it
    % into a single libnumbl_ops-backed function rather than per-statement
    % JS-side helper calls — this is exactly the workload Phase 3 of
    % the C-JIT was designed for.
    %
    % Written as a function file (not a script) because Octave 9 doesn't
    % accept local functions at the end of a script file. MATLAB, Octave,
    % and numbl all invoke the main function automatically.
    %
    % See benchmarks/tensor_ops_bench.md for full description + typical
    % results.

    N = 2000000;     % element count for each tensor
    trials = 50;     % how many times to repeat each kernel

    fprintf('N=%d, trials=%d\n', N, trials);
    fprintf('----------------------------------------\n');

    % Deterministic inputs (no RNG) so MATLAB / Octave / numbl agree
    % bitwise on the check values.
    t = linspace(-1.0, 1.0, N)';
    x = sin(3.1 .* t) .* 0.9;
    y = cos(2.7 .* t + 0.4) .* 0.8;

    % Warm-up: lands JIT specializations + C-JIT .node cache.
    warm = kernel_binary(x, y);
    warm = kernel_unary(x);
    warm = kernel_compare(x, y);
    warm = kernel_reduce(x);
    warm = kernel_chain(x, y);
    fprintf('warmup checks ok\n');

    % ── 1. Real binary element-wise (+, -, .*, ./, scalar mix) ─────────
    tic;
    for k = 1:trials
        r = kernel_binary(x, y);
    end
    t1 = toc;

    % ── 2. Real unary element-wise (exp, abs, sin, cos, tanh) ──────────
    tic;
    for k = 1:trials
        u = kernel_unary(x);
    end
    t2 = toc;

    % ── 3. Comparisons (>, <, ==) + reduction (sum) ────────────────────
    tic;
    cmp_acc = 0;
    for k = 1:trials
        cmp_acc = cmp_acc + kernel_compare(x, y);
    end
    t3 = toc;

    % ── 4. Reductions (sum, mean, max, min) ────────────────────────────
    tic;
    red_acc = 0;
    for k = 1:trials
        red_acc = red_acc + kernel_reduce(x);
    end
    t4 = toc;

    % ── 5. Chained pipeline (binary + unary + scalar mix + reduction) ──
    tic;
    chain_acc = 0;
    for k = 1:trials
        chain_acc = chain_acc + kernel_chain(x, y);
    end
    t5 = toc;

    total = t1 + t2 + t3 + t4 + t5;

    fprintf('Real binary elemwise:   %7.3f s\n', t1);
    fprintf('Real unary elemwise:    %7.3f s\n', t2);
    fprintf('Comparisons + reduce:   %7.3f s\n', t3);
    fprintf('Reductions:             %7.3f s\n', t4);
    fprintf('Chained pipeline:       %7.3f s\n', t5);
    fprintf('----------------------------------------\n');
    fprintf('elapsed = %.3f s\n', total);

    fprintf('\nCheck values (must match across runtimes):\n');
    fprintf('  sum(r)       = %.10g\n', sum(r));
    fprintf('  sum(u)       = %.10g\n', sum(u));
    fprintf('  cmp_acc      = %.10g\n', cmp_acc);
    fprintf('  red_acc      = %.10g\n', red_acc);
    fprintf('  chain_acc    = %.10g\n', chain_acc);
    disp('SUCCESS')
end

% ── Kernels ──────────────────────────────────────────────────────────────
%
% Each kernel is a single function so numbl's JIT can specialize it once
% and the C-JIT (--opt 2) can compile it into a libnumbl_ops-backed
% .node module. MATLAB and Octave just run them as plain element-wise
% expressions.

function r = kernel_binary(x, y)
    r = x + y;
    r = r - 0.5 .* x;
    r = r .* y + 3.0;
    r = r ./ (1 + abs(y));
end

function u = kernel_unary(x)
    u = exp(-x .* x);
    u = u .* cos(5 .* x);
    u = u + sin(x + 1) .* sin(x + 1);
    u = abs(u);
    u = tanh(u);
end

function s = kernel_compare(x, y)
    % Count how many positions satisfy (x > 0) AND (y < 0.5).
    c1 = x > 0;
    c2 = y < 0.5;
    s = sum(c1 .* c2);
end

function s = kernel_reduce(x)
    s = sum(x) + mean(x) + max(x) + min(x);
end

function s = kernel_chain(x, y)
    r = x .* y + 0.5;
    r = exp(-r .* r);
    r = r .* x;
    s = sum(r);
end
