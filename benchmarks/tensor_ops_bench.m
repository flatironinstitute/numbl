function tensor_ops_bench()
    % Tensor element-wise + reduction benchmark: numbl --opt 0/1/2 vs MATLAB
    % vs Octave. Companion to scalar_bench.m.
    %
    % The five kernels are inlined into `run_all` (instead of being called
    % as user functions) so numbl's outer-loop JIT has a pure tensor-op
    % body to specialize — no UserCall boundary per iteration. Under
    % --opt 1 the loops get JS-JIT'd; under --opt 2 they get C-JIT'd.
    %
    % Warm-up is a low-iteration-count call to `run_all` that lands the
    % loop specializations; the timed call reuses them.
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

    % Warm-up: run everything with a tiny trial count so the JIT has
    % landed specializations before the timed call.
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = run_all(x, y, 2);

    % Hot: real measurement.
    [t1, t2, t3, t4, t5, r, u, cmp_acc, red_acc, chain_acc] = ...
        run_all(x, y, trials);

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

function [t1, t2, t3, t4, t5, r, u, cmp_acc, red_acc, chain_acc] = run_all(x, y, trials)
    % All five kernels inlined. Each outer loop has no UserCall in its
    % body, so the loop-JIT can specialize the full loop as a single
    % unit under both --opt 1 and --opt 2.

    % Initialize outputs so the function is well-defined even if
    % `trials == 0`. For the timed calls (trials >= 1) these are
    % overwritten by the loop bodies on the first iteration.
    r = x;
    u = x;

    % ── 1. Real binary element-wise (+, -, .*, ./, scalar mix) ─────────
    tic;
    for k = 1:trials
        r = x + y;
        r = r - 0.5 .* x;
        r = r .* y + 3.0;
        r = r ./ (1 + abs(y));
    end
    t1 = toc;

    % ── 2. Real unary element-wise (exp, abs, sin, cos, tanh) ──────────
    tic;
    for k = 1:trials
        u = exp(-x .* x);
        u = u .* cos(5 .* x);
        u = u + sin(x + 1) .* sin(x + 1);
        u = abs(u);
        u = tanh(u);
    end
    t2 = toc;

    % ── 3. Comparisons (>, <) + reduction (sum) ────────────────────────
    tic;
    cmp_acc = 0;
    for k = 1:trials
        c1 = x > 0;
        c2 = y < 0.5;
        cmp_acc = cmp_acc + sum(c1 .* c2);
    end
    t3 = toc;

    % ── 4. Reductions (sum, mean, max, min) ────────────────────────────
    tic;
    red_acc = 0;
    for k = 1:trials
        red_acc = red_acc + (sum(x) + mean(x) + max(x) + min(x));
    end
    t4 = toc;

    % ── 5. Chained pipeline (binary + unary + scalar mix + reduction) ──
    tic;
    chain_acc = 0;
    for k = 1:trials
        r2 = x .* y + 0.5;
        r2 = exp(-r2 .* r2);
        r2 = r2 .* x;
        chain_acc = chain_acc + sum(r2);
    end
    t5 = toc;
end
