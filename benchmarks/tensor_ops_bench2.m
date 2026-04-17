function tensor_ops_bench2()
    % Benchmark for extended fusion: single-expression fusion, inline-
    % reduction fusion, and two-argument element-wise builtins.
    % Companion to tensor_ops_bench.m.
    %
    % Kernels exercise patterns the original benchmark did not cover:
    %   - Single assignments with deep expression trees
    %   - Reductions of inline element-wise expressions
    %   - Two-arg element-wise builtins (max, min, hypot, atan2, mod)
    %
    % See benchmarks/tensor_ops_bench2.md for full description.

    N = 2000000;
    trials = 50;

    fprintf('N=%d, trials=%d\n', N, trials);
    fprintf('----------------------------------------\n');

    t = linspace(-1.0, 1.0, N)';
    x = sin(3.1 .* t) .* 0.9;
    y = cos(2.7 .* t + 0.4) .* 0.8;

    % Warm-up
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = run_all(x, y, 2);

    % Hot
    [t1, t2, t3, t4, t5, t6, u1, u2, u3, ir_acc] = run_all(x, y, trials);

    total = t1 + t2 + t3 + t4 + t5 + t6;

    fprintf('Single-expr Gaussian:   %7.3f s\n', t1);
    fprintf('Single-expr nested:     %7.3f s\n', t2);
    fprintf('Inline reduction:       %7.3f s\n', t3);
    fprintf('Inline accum reduction: %7.3f s\n', t4);
    fprintf('Binary builtins:        %7.3f s\n', t5);
    fprintf('Clamp + distance:       %7.3f s\n', t6);
    fprintf('----------------------------------------\n');
    fprintf('elapsed = %.3f s\n', total);

    fprintf('\nCheck values (must match across runtimes):\n');
    fprintf('  sum(u1)      = %.10g\n', sum(u1));
    fprintf('  sum(u2)      = %.10g\n', sum(u2));
    fprintf('  sum(u3)      = %.10g\n', sum(u3));
    fprintf('  ir_acc       = %.10g\n', ir_acc);
    disp('SUCCESS')
end

function [t1, t2, t3, t4, t5, t6, u1, u2, u3, ir_acc] = run_all(x, y, trials)
    u1 = x;
    u2 = x;
    u3 = x;

    % -- 1. Single-expression fusion: Gaussian kernel ----------------------
    % One assignment with 3 tensor ops (ElemMul, Negate, exp).
    tic;
    for k = 1:trials
        u1 = exp(-x .* x);
    end
    t1 = toc;

    % -- 2. Single-expression fusion: deep nested unaries ------------------
    % One assignment with nested unary calls + arithmetic.
    tic;
    for k = 1:trials
        u2 = tanh(abs(sin(x + 1) .* sin(x + 1) + cos(y .* 2)));
    end
    t2 = toc;

    % -- 3. Inline reduction: sum of element-wise expression ---------------
    % No named intermediate -- the reduction argument is an expression.
    tic;
    ir_acc = 0;
    for k = 1:trials
        ir_acc = sum(x .* y + 0.5);
    end
    t3 = toc;

    % -- 4. Inline accumulate reduction: acc += sum(expr) ------------------
    tic;
    ir_acc = 0;
    for k = 1:trials
        ir_acc = ir_acc + sum(exp(-x .* x));
    end
    t4 = toc;

    % -- 5. Two-arg element-wise builtins: max, min, atan2, hypot ----------
    % Chain of two-arg tensor builtins that should fuse into one loop.
    tic;
    for k = 1:trials
        u3 = max(x, y);
        u3 = u3 + atan2(y, x);
        u3 = u3 .* hypot(x, y);
    end
    t5 = toc;

    % -- 6. Clamp + Euclidean distance pipeline ----------------------------
    % Clamp x to [-0.5, 0.5], then compute Euclidean distance to y.
    % Exercises min, max, hypot in a chain with arithmetic.
    tic;
    for k = 1:trials
        u3 = max(min(x, 0.5), -0.5);
        u3 = hypot(u3 - y, x .* y);
    end
    t6 = toc;
end
