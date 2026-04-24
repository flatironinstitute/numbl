function bench_03_scalar_builtins()
    % Level 03 -- scalar math builtins in the body.
    %
    % Adds calls to whitelisted scalar builtins (sin, cos, exp, sqrt).
    % The kernel's fused-scalar emitter routes these through each
    % IBuiltin's jitEmitC so they lower to the corresponding C library
    % functions.
    %
    % The result is a non-trivial sum we can compare across runtimes.

    n = 1000000;

    % Warmup
    s = 0;
    for i = 1:1000
        s = s + sin(i * 0.1) * cos(i * 0.2) + sqrt(i * 0.01);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        s = s + sin(i * 0.1) * cos(i * 0.2) + sqrt(i * 0.01);
    end
    t_elapsed = toc(t0);

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=scalar_builtins n=%d elapsed=%.6f per_iter_ns=%.2f s=%.10e\n', ...
        n, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
