function bench_02_scalar_locals()
    % Level 02 -- multiple body assigns with loop-local scalars.
    %
    % Introduces two new pieces vs level 01:
    %   - A loop-local scalar `x` that's written inside the loop but not
    %     read outside — the kernel should treat it as a local, but
    %     MATLAB semantics leave its post-loop value in env.
    %   - A chain of two Assigns (x = ..., s = ...) in the body, so the
    %     second assign reads the freshly-written local.
    %
    % The closed-form answer is:
    %   s = sum_{i=1..n} (0.1*i)^2 = 0.01 * n*(n+1)*(2n+1)/6

    n = 1000000;

    % Warmup
    s = 0;
    for i = 1:1000
        x = i * 0.1;
        s = s + x * x;
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        x = i * 0.1;
        s = s + x * x;
    end
    t_elapsed = toc(t0);

    expected = 0.01 * n * (n + 1) * (2 * n + 1) / 6;
    rel_err = abs(s - expected) / abs(expected);
    assert(rel_err < 1e-10, ...
        sprintf('scalar_locals: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=scalar_locals n=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e x_last=%.6e\n', ...
        n, t_elapsed, per_iter_ns, s, x);
    disp('SUCCESS')
end
