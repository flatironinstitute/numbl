function bench_06_inline_elemwise()
    % Level 06 -- inline elementwise + reduction inside the loop.
    %
    % Each iteration builds a small tensor `c = a.*b + i*0.001` and
    % reduces it with sum. A naive implementation materializes `c`
    % every iteration (huge per-iter malloc + dispatch cost). The
    % whole-loop C JIT should fuse the elementwise op directly into
    % the sum accumulator -- no temp tensor allocated, one inner C
    % loop over the tensor length.
    %
    % Expected pattern in the emitted kernel:
    %   for (int64_t __iv = lo; ...) {
    %       double __sum_c = 0.0;
    %       for (int64_t __j = 0; __j < n_a; __j++) {
    %           __sum_c += v_a[__j] * v_b[__j] + v_i * 0.001;
    %       }
    %       v_s = v_s + __sum_c;
    %   }
    %
    % `a` and `b` are kept short (k=32) so the inner loop is a
    % realistic stand-in for the chunkie inner kernel shape.
    %
    % Reference: sum over i of sum(a.*b + i*0.001)
    %         = n*dot(a,b) + 0.001*k*(n*(n+1)/2)

    n = 200000;
    k = 32;
    a = linspace(0.1, 1.0, k);
    b = linspace(-0.5, 0.5, k);

    % Warmup
    s = 0;
    for i = 1:1000
        c = a .* b + i * 0.001;
        s = s + sum(c);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        c = a .* b + i * 0.001;
        s = s + sum(c);
    end
    t_elapsed = toc(t0);

    expected = n * dot(a, b) + 0.001 * k * (n * (n + 1) / 2);
    assert(abs(s - expected) / abs(expected) < 1e-10, ...
        sprintf('level 06: s=%.6e expected=%.6e', s, expected));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=inline_elemwise n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
