% Complex qr (plain and column-pivoted) and lu against MATLAB values.
% Under vitest these run on the ts-lapack bridge (the browser path).

A = [2+1i 1-1i 3; -1 4+2i 1i; 0.5i 2 -2+1i; 1+1i -3i 1];
tol = 1e-12;

% qr, economy and full
R_exp = [-2.87228132326901 2.08893187146837+3.13339780720256i -2.61116483933547+1.39262124764558i;0 -4.56269457866531 -0.458261900913984-0.19924430474521i;0 0 2.6443755367991];
[Q, R] = qr(A, 0);
assert(isequal(size(Q), [4 3]) && isequal(size(R), [3 3]));
assert(norm(R - R_exp) < tol*norm(A));
assert(norm(Q*R - A) < tol*norm(A));
assert(norm(Q'*Q - eye(3)) < tol);
assert(isequal(R, qr(A, 0)));
[Q, R] = qr(A);
assert(isequal(size(Q), [4 4]) && isequal(size(R), [4 3]));
assert(norm(Q*R - A) < tol*norm(A));
assert(norm(Q'*Q - eye(4)) < tol);
assert(norm(R(1:3, :) - R_exp) < tol*norm(A) && norm(R(4, :)) == 0);
[Q, R] = qr(A.');
assert(norm(Q*R - A.') < tol*norm(A) && norm(tril(R, -1)) == 0);

% column-pivoted qr
Rp_exp = [-5.91607978309962 -0.169030850945703-2.02837021134844i 1.01418510567422-1.52127765851133i;0 3.44342022662684 1.33173240090026+0.489546490050563i;0 0 -1.70116898149903];
[Q, R, p] = qr(A, 0);
assert(isequal(p, [2 3 1]));
assert(norm(R - Rp_exp) < tol*norm(A));
assert(norm(Q*R - A(:, p)) < tol*norm(A));
[Q, R, P] = qr(A);
E = eye(3);
assert(isequal(P, E(:, [2 3 1])));
assert(norm(Q*R - A*P) < tol*norm(A));
[~, ~, p] = qr([1 3i 2; 1i 1 -2+1i; 0 1 1], 0);
assert(isequal(p, [2 3 1]));

% lu, pivoting by |re| + |im| as in LAPACK izamax
C = [3 1 1i; 2+2i 1i 0; -1 2 1];
U_exp = [2+2i 0+1i 0;0 2.25+0.25i 1;0 0 -0.0731707317073171+1.34146341463415i];
[L, U, p] = lu(C, 'vector');
assert(isequal(p, [2 3 1]));
assert(norm(U - U_exp) < tol*norm(C));
assert(norm(L*U - C(p, :)) < tol*norm(C));
[L, U, P] = lu(C);
assert(norm(L*U - P*C) < tol*norm(C));
[L, U] = lu(C);
assert(norm(L*U - C) < tol*norm(C));
U_exp = [2+1i 1-1i 3;0 4.2+1.4i 1.2+0.4i;0 0 -2.78571428571429+0.428571428571429i];
[L, U, p] = lu(A, 'vector');
assert(isequal(p, [1 2 3 4]) && isequal(size(L), [4 3]));
assert(norm(U - U_exp) < tol*norm(A));
assert(norm(L*U - A) < tol*norm(A));
At = A.';
[L, U, p] = lu(At, 'vector');
assert(isequal(size(U), [3 4]));
assert(norm(L*U - At(p, :)) < tol*norm(A));

disp('SUCCESS')
