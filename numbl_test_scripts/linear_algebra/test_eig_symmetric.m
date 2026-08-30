% Symmetric (and Hermitian) input must take the symmetric eigensolver, which
% guarantees what the general dgeev path cannot: real eigenvalues in ascending
% order and an eigenvector basis that stays orthonormal across repeated or
% tightly clustered eigenvalues.

n = 40;
v = (1:n)' - (n + 1) / 2;  v = v / sqrt(v' * v);
w = mod((1:n)', 7) - 3;    w = w / sqrt(w' * w);
Q = (eye(n) - 2 * (v * v')) * (eye(n) - 2 * (w * w'));

for k = 1:3
    d = (1:n)';
    if k == 1
        d(10) = 12;  d(11) = 12;                  % exactly repeated pair
    elseif k == 2
        d(20) = 25;  d(21) = 25 + 1e-11;          % pair split by 1e-11
    else
        d(5) = 7;  d(6) = 7;  d(7) = 7;           % triple degenerate
    end
    S = Q * diag(d) * Q';
    S = (S + S') / 2;
    assert(isequal(S, S'));

    [V, D] = eig(S);
    e = diag(D);
    assert(max(abs(imag(e))) == 0);
    assert(all(diff(e) >= 0));
    assert(max(max(abs(V' * V - eye(n)))) < 1e-12);
    assert(max(max(abs(S * V - V * D))) < 1e-10 * norm(S));
    assert(max(abs(e - sort(d))) < 1e-9 * norm(S));

    % Eigenvalues-only agrees with the two-output form.
    assert(max(abs(eig(S) - e)) < 1e-12 * norm(S));
end

% Hermitian input takes the same path: real eigenvalues, unitary basis.
m = 10;
Ar = zeros(m);  Ai = zeros(m);
for i = 1:m
    for j = 1:m
        Ar(i, j) = cos(i * j);
        Ai(i, j) = sin(i + 2 * j);
    end
end
H = (Ar + Ar') / 2 + 1i * (Ai - Ai') / 2;
assert(isequal(H, H'));
[Vh, Dh] = eig(H);
eh = diag(Dh);
assert(max(abs(imag(eh))) == 0);
assert(all(diff(real(eh)) >= 0));
assert(max(max(abs(Vh' * Vh - eye(m)))) < 1e-12);
assert(max(max(abs(H * Vh - Vh * Dh))) < 1e-10 * norm(H));

% A Hermitian matrix with a repeated eigenvalue still gives a unitary basis.
Hd = Vh * diag([1 1 1 2 2 3 4 5 6 7]) * Vh';
Hd = (Hd + Hd') / 2;
[Vd, Dd] = eig(Hd);
assert(max(max(abs(Vd' * Vd - eye(m)))) < 1e-12);
assert(max(abs(diag(Dd) - [1 1 1 2 2 3 4 5 6 7]')) < 1e-10);

% Three outputs: for a symmetric matrix the left eigenvectors coincide with
% the right ones, so W'*A = D*W' must hold for the returned W.
[Vt, Dt, Wt] = eig([4 1 0; 1 3 1; 0 1 2]);
assert(max(max(abs(Wt' * [4 1 0; 1 3 1; 0 1 2] - Dt * Wt'))) < 1e-12);

% The general path is still reachable and still correct. A nonsymmetric
% matrix keeps its complex eigenvalues and its non-orthogonal eigenvectors.
N = [0 -1 0; 1 0 0; 0 0 2];
en = eig(N);
assert(max(abs(imag(en))) > 0.5);
G = [1 2 3; 4 5 6; 7 8 10];
[Vg, Dg] = eig(G);
assert(max(max(abs(G * Vg - Vg * Dg))) < 1e-10 * norm(G));
assert(max(max(abs(Vg' * Vg - eye(3)))) > 1e-3);

% Symmetry is tested exactly, as MATLAB's isequal(A, A') does: a matrix that
% is symmetric only to within rounding takes the general path.
P = [2 1; 1 + 1e-15 3];
assert(~isequal(P, P'));
ep = sort(real(eig(P)));
assert(max(abs(ep - sort(real(eig([2 1; 1 3]))))) < 1e-7);

disp('SUCCESS')
