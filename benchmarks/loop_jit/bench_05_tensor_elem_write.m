function bench_05_tensor_elem_write()
    % Level 05 -- write a tensor element per iteration.
    %
    % Preallocated output `y`, and each iteration writes y(i) from a
    % scalar expression. The kernel must:
    %   - accept `y` as a `double *` (inout tensor — we only touch
    %     element i, other elements are preserved)
    %   - emit `v_y[(int64_t)(v_i) - 1] = <rhs>;`
    %
    % The result `y` is checked against an elementwise reference.

    n = 1000000;
    y = zeros(1, n);

    % Warmup
    for i = 1:1000
        y(i) = sin(i * 0.01);
    end

    y = zeros(1, n);
    t0 = tic;
    for i = 1:n
        y(i) = sin(i * 0.01);
    end
    t_elapsed = toc(t0);

    % Spot-check a few entries
    assert(abs(y(1) - sin(0.01)) < 1e-12);
    assert(abs(y(n) - sin(n * 0.01)) < 1e-12);
    assert(abs(y(n/2) - sin((n/2) * 0.01)) < 1e-12);

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=tensor_elem_write n=%d elapsed=%.6f per_iter_ns=%.2f y_sum=%.6e\n', ...
        n, t_elapsed, per_iter_ns, sum(y));
    disp('SUCCESS')
end
