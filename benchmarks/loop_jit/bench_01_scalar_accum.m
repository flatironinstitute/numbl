function bench_01_scalar_accum()
    % Level 01 -- scalar accumulator driven by the loop variable.
    %
    % Simplest meaningful loop: accumulate sqrt(i) across iterations.
    % We use sqrt(i) rather than i directly so that the sum has no
    % polynomial closed form -- GCC's scalar-evolution pass (which under
    % -ffast-math will otherwise rewrite `s = s + i` into `n*(n+1)/2`)
    % bails out on non-polynomial induction.
    %
    % This exercises:
    %   - for i = 1:n  range-form for-loop
    %   - a live scalar (s) that exists before, during, and after the loop
    %   - one scalar sqrt + add per iteration depending on the loop variable
    %
    % Under --opt e2 today the interpreter walks the loop body every
    % iteration, paying ~70-100 ns / iter just for AST dispatch plus the
    % scalar work. Goal: detect this pattern, emit a C kernel for the
    % whole loop, and match MATLAB.

    n = 10000000;

    % Warmup (also ensures s / i are typed consistently before timing)
    s = 0;
    for i = 1:1000
        s = s + sqrt(i);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        s = s + sqrt(i);
    end
    t_elapsed = toc(t0);

    expected = sum(sqrt(1:n));
    rel_err = abs(s - expected) / expected;
    assert(rel_err < 1e-10, ...
        sprintf('scalar_accum: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=scalar_accum n=%d elapsed=%.6f per_iter_ns=%.2f s=%.10e\n', ...
        n, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
