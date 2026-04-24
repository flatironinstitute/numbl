function bench_04_tensor_elem_read()
    % Level 04 -- read an element of a tensor input inside the loop.
    %
    % This is the first level that depends on tensor data. `x` is a
    % precomputed vector; the loop accumulates x(i) across iterations.
    % The kernel needs to:
    %   - accept `x` as a `const double *` kernel param
    %   - emit `v_x_ptr[__iv - 1]` for the `x(i)` index expression
    %
    % The result is exact: sum of x is independent of floating-point
    % order for integer / clean-fraction values.

    n = 1000000;
    x = (1:n) * 0.001;          % row vector 1..n

    % Warmup
    s = 0;
    for i = 1:1000
        s = s + x(i);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        s = s + x(i);
    end
    t_elapsed = toc(t0);

    expected = 0.001 * n * (n + 1) / 2;
    rel_err = abs(s - expected) / abs(expected);
    assert(rel_err < 1e-10, ...
        sprintf('tensor_elem_read: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=tensor_elem_read n=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
