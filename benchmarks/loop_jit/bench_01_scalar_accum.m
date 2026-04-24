function bench_01_scalar_accum()
    % Level 01 -- scalar accumulator driven by the loop variable.
    %
    % Simplest meaningful loop: accumulate i across iterations. The
    % result s = n*(n+1)/2 is observed and asserted, so no compiler /
    % JIT can elide the loop.
    %
    % This exercises:
    %   - for i = 1:n  range-form for-loop
    %   - a live scalar (s) that exists before, during, and after the loop
    %   - one scalar add per iteration depending on the loop variable
    %
    % Under --opt e2 today the interpreter walks the loop body every
    % iteration, paying ~70-100 ns / iter just for AST dispatch plus the
    % scalar add. Goal: detect this pattern, emit a C kernel for the
    % whole loop, and match MATLAB.

    n = 10000000;

    % Warmup (also ensures s / i are typed consistently before timing)
    s = 0;
    for i = 1:1000
        s = s + i;
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        s = s + i;
    end
    t_elapsed = toc(t0);

    expected = n * (n + 1) / 2;
    assert(s == expected, ...
        sprintf('scalar_accum: s=%g expected=%g', s, expected));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=scalar_accum n=%d elapsed=%.6f per_iter_ns=%.2f s=%.0f\n', ...
        n, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
