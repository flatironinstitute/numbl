% Test gmres builtin

% Test 1: Simple SPD system
A = [4 1 0; 1 3 1; 0 1 2];
b = [1; 2; 3];
x = gmres(A, b, [], 1e-10);
assert(norm(A*x - b) / norm(b) < 1e-8, 'gmres: basic solve failed');

% Test 2: [x, flag] output
[x2, flag2] = gmres(A, b, [], 1e-10);
assert(flag2 == 0, 'gmres: should converge for SPD system');
assert(norm(A*x2 - b) / norm(b) < 1e-8, 'gmres: solution inaccurate');

% Test 3: [x, flag, relres]
[x3, flag3, relres3] = gmres(A, b, [], 1e-10);
assert(flag3 == 0, 'gmres: flag should be 0');
assert(relres3 < 1e-8, 'gmres: relres should be small');

% Test 4: [x, flag, relres, iter]
[x4, flag4, relres4, iter4] = gmres(A, b, [], 1e-10);
assert(length(iter4) == 2, 'gmres: iter should have 2 elements');

% Test 5: [x, flag, relres, iter, resvec]
[x5, flag5, relres5, iter5, resvec5] = gmres(A, b, [], 1e-10);
assert(length(resvec5) > 1, 'gmres: resvec should have multiple entries');

% Test 6: Larger tridiagonal system with restart
n = 20;
A2 = zeros(n);
for i = 1:n
    A2(i,i) = 4;
    if i > 1, A2(i,i-1) = -1; end
    if i < n, A2(i,i+1) = -1; end
end
b2 = ones(n, 1);
[x6, flag6] = gmres(A2, b2, 5, 1e-10, 20);
assert(flag6 == 0, 'gmres: should converge with restart');
assert(norm(A2*x6 - b2) / norm(b2) < 1e-8, 'gmres: residual too large with restart');

% Test 7: With Jacobi preconditioner
M = diag(diag(A2));
[x7, flag7] = gmres(A2, b2, 5, 1e-10, 20, M);
assert(flag7 == 0, 'gmres: should converge with Jacobi preconditioner');
assert(norm(A2*x7 - b2) / norm(b2) < 1e-8, 'gmres: preconditioned solve inaccurate');

% Test 8: With M1, M2 split preconditioner
[L, U] = lu(M);
[x8, flag8] = gmres(A2, b2, 5, 1e-10, 20, L, U);
assert(flag8 == 0, 'gmres: should converge with LU preconditioner');

% Test 9: With initial guess (near-exact solution)
x_exact = A \ b;
x0 = x_exact + 1e-8 * ones(3, 1);
[x9, flag9] = gmres(A, b, [], 1e-10, 10, [], [], x0);
assert(flag9 == 0, 'gmres: should converge quickly with good guess');

% Test 10: Compare with backslash
x_bs = A2 \ b2;
[x10, flag10] = gmres(A2, b2, [], 1e-12, 100);
assert(flag10 == 0, 'gmres: should converge');
assert(norm(x10 - x_bs) / norm(x_bs) < 1e-6, 'gmres: should match backslash');

% Test 11: Complex system
Ac = [4+1i 1; 1 3-1i];
bc = [1+2i; 3-1i];
[x11, flag11] = gmres(Ac, bc, [], 1e-10);
assert(flag11 == 0, 'gmres: complex solve should converge');
x_exact_c = Ac \ bc;
assert(norm(x11 - x_exact_c) / norm(x_exact_c) < 1e-8, 'gmres: complex solve inaccurate');

% Test 12: Larger complex system
n2 = 15;
Ac2 = zeros(n2);
for i = 1:n2
    Ac2(i,i) = 4 + 0.5i;
    if i > 1, Ac2(i,i-1) = -1 + 0.1i; end
    if i < n2, Ac2(i,i+1) = -1 - 0.1i; end
end
bc2 = ones(n2, 1) + 0.5i * ones(n2, 1);
[x12, flag12] = gmres(Ac2, bc2, 5, 1e-10, 20);
assert(flag12 == 0, 'gmres: complex restart should converge');
assert(norm(Ac2*x12 - bc2) / norm(bc2) < 1e-8, 'gmres: complex restart residual too large');

% Test 13: Complex system with all outputs
[x13, flag13, relres13, iter13, resvec13] = gmres(Ac, bc, [], 1e-10);
assert(flag13 == 0, 'gmres: complex all-output flag');
assert(relres13 < 1e-8, 'gmres: complex relres');
assert(length(iter13) == 2, 'gmres: complex iter length');
assert(length(resvec13) > 1, 'gmres: complex resvec length');

disp('SUCCESS');
