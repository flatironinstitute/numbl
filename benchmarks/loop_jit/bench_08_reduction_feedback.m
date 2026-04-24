function bench_08_reduction_feedback()
    % Level 08 -- scalar reduction feeds back into a later tensor_local.
    %
    % Body:
    %   r = a + i*0.01;   % tensor_local (depends on loop var)
    %   q = sum(r);       % reduction -> scalar q
    %   t = b*q + i;      % tensor_local depending on the reduction scalar
    %   s = s + sum(t);   % final reduction into inout scalar
    %
    % Tests: two separate reduction loops in the same iteration, with
    % the first reduction's scalar result feeding an expression inside
    % the second tensor_local. No materialization of r or t; each
    % reduction runs its own inlined inner loop.

    n = 200000;
    k = 32;
    a = linspace(0.1, 1.0, k);
    b = linspace(-0.5, 0.5, k);

    % Warmup
    s = 0;
    for i = 1:1000
        r = a + i * 0.01;
        q = sum(r);
        t = b * q + i;
        s = s + sum(t);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        r = a + i * 0.01;
        q = sum(r);
        t = b * q + i;
        s = s + sum(t);
    end
    t_elapsed = toc(t0);

    % Expected:
    %   sum_r(i) = sum(a) + 0.01*i*k
    %   q = sum_r(i)
    %   sum_t(i) = q*sum(b) + i*k
    %   total s = sum over i=1..n of (q*sum(b) + i*k)
    sA = sum(a);
    sB = sum(b);
    expected = 0;
    for i = 1:n
        qi = sA + 0.01 * i * k;
        expected = expected + qi * sB + i * k;
    end
    assert(abs(s - expected) / max(1, abs(expected)) < 1e-8, ...
        sprintf('level 08: s=%.6e expected=%.6e', s, expected));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=reduction_feedback n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
