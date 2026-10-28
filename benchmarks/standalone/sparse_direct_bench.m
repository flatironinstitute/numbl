% sparse_direct_bench — sparse direct solvers on the implicit plate workload.
%
% Assembles the Kirchhoff-Love plate operator on the interior nodes of an
% nx x ny grid (column-major node ordering, so the matrix is banded with
% half-bandwidth mx = nx - 2), factors it once with chol, and runs the two
% triangular solves per time step that an implicit scheme needs.
%
% Run with:  npx tsx src/cli.ts run benchmarks/standalone/sparse_direct_bench.m --no-native
%            matlab -batch "run('benchmarks/standalone/sparse_direct_bench.m')"

grids = [61 41 900; 121 81 200];   % nx, ny, nsteps
for g = 1:size(grids, 1)
    nx = grids(g, 1); ny = grids(g, 2); nsteps = grids(g, 3);
    mx = nx - 2; my = ny - 2; m = mx * my;
    dx = 1 / (nx - 1); dy = 1 / (ny - 1);
    kappa = 1; sigma0 = 1; sigma1 = 0.005; theta = 0.5; dt = dx^2;
    fprintf('--- grid %d x %d, m = %d, nsteps = %d ---\n', nx, ny, m, nsteps);

    t = tic;
    e = ones(mx, 1); D2x = spdiags([e -2*e e], -1:1, mx, mx) / dx^2;
    e = ones(my, 1); D2y = spdiags([e -2*e e], -1:1, my, my) / dy^2;
    L = kron(speye(my), D2x) + kron(D2y, speye(mx));
    B = kron(speye(my), D2x*D2x) + 2*kron(D2y, D2x) + kron(D2y*D2y, speye(mx));
    A = (1 + sigma0*dt) * speye(m) - sigma1*dt*L + theta*kappa^2*dt^2 * B;
    fprintf('%-32s %9.2f ms   (nnz(A) = %d)\n', 'assembly (kron)', 1000*toc(t), nnz(A));

    t = tic;
    R = chol(A);
    fprintf('%-32s %9.2f ms   (nnz(R) = %d)\n', 'chol(A)', 1000*toc(t), nnz(R));

    Rt = R';
    w = zeros(m, 1); v = zeros(m, 1);
    rng(0); f = randn(m, 1);
    t = tic;
    for n = 1:nsteps
        rhs = w + dt*v + dt^2*f - theta*kappa^2*dt^2*(B*w);
        wn = R \ (Rt \ rhs);
        v = (wn - w) / dt;
        w = wn;
    end
    el = 1000*toc(t);
    fprintf('%-32s %9.2f ms   (%.3f ms per step)\n', 'time loop (2 tri solves + B*w)', el, el / nsteps);

    b = randn(m, 1);
    t = tic;
    x = A \ b;
    fprintf('%-32s %9.2f ms   (residual %.1e)\n', 'A \\ b (no stored factor)', 1000*toc(t), norm(A*x - b) / norm(b));

    t = tic;
    [Lf, Uf, P] = lu(A);
    fprintf('%-32s %9.2f ms   (nnz(L)+nnz(U) = %d)\n', 'lu(A)', 1000*toc(t), nnz(Lf) + nnz(Uf));

    t = tic;
    dA = decomposition(A, 'chol');
    for n = 1:nsteps
        x = dA \ b;
    end
    el = 1000*toc(t);
    fprintf('%-32s %9.2f ms   (%.3f ms per solve)\n', 'decomposition + nsteps solves', el, el / nsteps);
end
