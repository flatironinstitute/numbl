% eig(A, B) must solve the generalized problem A*v = lambda*B*v, not quietly
% return eig(A). The symmetric-definite pencil is reduced through the Cholesky
% factor of B and normalized so that V'*B*V = I; everything else goes through
% the QZ factorization.

A = [4 1 0; 1 3 1; 0 1 2];
B = [2 0 0; 0 5 0; 0 0 9];

lam = eig(A, B);
assert(numel(lam) == 3);
assert(max(abs(imag(lam))) == 0);
assert(all(diff(lam) >= 0));
for k = 1:3
    assert(abs(det(A - lam(k) * B)) < 1e-9);
end

% The whole point: the answer differs from the standard problem.
assert(max(abs(sort(lam) - sort(eig(A)))) > 1e-3);

[V, D] = eig(A, B);
assert(max(max(abs(A * V - B * V * D))) < 1e-12);
assert(max(max(abs(V' * B * V - eye(3)))) < 1e-12);

% Third output: left eigenvectors, W'*A = D*W'*B.
[V2, D2, W2] = eig(A, B);
assert(max(max(abs(W2' * A - D2 * W2' * B))) < 1e-12);

% Output form and algorithm options.
[Vv, dv] = eig(A, B, 'vector');
assert(size(dv, 1) == 3 && size(dv, 2) == 1);
assert(max(abs(dv - lam)) < 1e-12);
assert(max(max(abs(A * Vv - B * Vv * diag(dv)))) < 1e-12);
lc = eig(A, B, 'chol');
assert(max(abs(sort(lc) - lam)) < 1e-12);

% Hermitian-definite pencil.
Ah = [2 1i; -1i 3];
Bh = [2 0; 0 4];
[Vh, Dh] = eig(Ah, Bh);
assert(max(abs(imag(diag(Dh)))) == 0);
assert(max(max(abs(Ah * Vh - Bh * Vh * Dh))) < 1e-12);
assert(max(max(abs(Vh' * Bh * Vh - eye(2)))) < 1e-12);

% Repeated eigenvalues in a symmetric-definite pencil still give a B-
% orthonormal basis.
Ar = diag([1 1 1 4]);
Br = diag([2 2 2 8]);
[Vr, Dr] = eig(Ar, Br);
assert(max(abs(diag(Dr) - [0.5; 0.5; 0.5; 0.5])) < 1e-12);
assert(max(max(abs(Vr' * Br * Vr - eye(4)))) < 1e-12);

% General (non symmetric-definite) pencil: the QZ path. numbl needs the native
% LAPACK addon for that one, so skip these checks where it is unavailable —
% the alternative there is an error, never a wrong answer.
An = [1 2 3; 4 5 6; 7 8 10];
Bn = [2 1 0; 0 3 1; 0 0 4];
haveQz = true;
try
    lg = eig(An, Bn);
catch
    haveQz = false;
end
if haveQz
    assert(numel(lg) == 3);
    for k = 1:3
        assert(abs(det(An - lg(k) * Bn)) < 1e-8);
    end
    [Vn, Dn] = eig(An, Bn);
    assert(max(max(abs(An * Vn - Bn * Vn * Dn))) < 1e-10);
    % QZ eigenvectors are normalized so the largest component has magnitude 1.
    assert(max(abs(max(abs(Vn), [], 1) - 1)) < 1e-12);
    [Vq, Dq, Wq] = eig(An, Bn);
    assert(max(max(abs(Wq' * An - Dq * Wq' * Bn))) < 1e-10);
    % An indefinite B also has to work through QZ.
    Ai = [1 2; 3 4];
    Bi = [1 0; 0 -1];
    [Vi, Di] = eig(Ai, Bi);
    assert(max(max(abs(Ai * Vi - Bi * Vi * Di))) < 1e-12);
end

% Mismatched sizes are an error, not a silently dropped argument.
bad = false;
try
    eig([1 2; 3 4], [1 2 3; 4 5 6; 7 8 9]);
catch
    bad = true;
end
assert(bad);

disp('SUCCESS')
