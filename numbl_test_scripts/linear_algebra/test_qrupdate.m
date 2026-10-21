% qrupdate: rank-1 update of a full QR factorization, Q1*R1 = Q*R + u*v'

% matches MATLAB's sign convention
[Q1, R1] = qrupdate(eye(3), triu(magic(3)), [1; 2; 3], [0; 1; 0]);
R_exp = [8 2 6; 0 7.615773105863909 7.221853807284738; 0 0 -0.919145030018058];
Q_exp = [1 0 0; 0 0.919145030018058 -0.393919298579168; 0 0.393919298579168 0.919145030018058];
assert(norm(R1 - R_exp) < 1e-12);
assert(norm(Q1 - Q_exp) < 1e-12);

% tall, real
A = magic(5); A = A(:, 1:3);
[Q, R] = qr(A);
u = (1:5)'; v = [1; -1; 2];
[Q1, R1] = qrupdate(Q, R, u, v);
assert(isequal(size(Q1), [5 5]) && isequal(size(R1), [5 3]));
assert(isreal(Q1) && isreal(R1));
assert(norm(Q1*R1 - (A + u*v')) < 1e-12*norm(A));
assert(norm(Q1'*Q1 - eye(5)) < 1e-12);
assert(norm(tril(R1, -1)) == 0);
assert(abs(abs(R1(1,1)) - norm(A(:,1) + u*v(1))) < 1e-12);

% wide, real
A = magic(4); A = A(1:3, :);
[Q, R] = qr(A);
[Q1, R1] = qrupdate(Q, R, [1; 2; 3], ones(4, 1));
assert(norm(Q1*R1 - (A + [1; 2; 3]*ones(1, 4))) < 1e-12*norm(A));
assert(norm(tril(R1, -1)) == 0);

% complex (start from Q = I, so no complex qr needed)
R = triu([2+1i 1-1i 3; 0 1+2i -1i; 0 0 4-1i; 0 0 0]);
u = [1; 1i; -2; 0.5+0.5i]; v = [1i; 2; -1+1i];
[Q1, R1] = qrupdate(eye(4), R, u, v);
assert(~isreal(Q1) && ~isreal(R1));
assert(norm(Q1*R1 - (R + u*v')) < 1e-12*norm(R));
assert(norm(Q1'*Q1 - eye(4)) < 1e-12);
assert(norm(tril(R1, -1)) == 0);

% FLAM id.m use: swap two columns of a triangular factor
R = triu(magic(4)) + eye(4);
i = 2; k = 4;
[~, R1] = qrupdate(eye(k), R, R(:,k) - R(:,i), ((1:k == i) - (1:k == k))');
P = R; P(:, [i k]) = P(:, [k i]);
assert(norm(tril(R1, -1)) == 0);
assert(norm(abs(R1(1,:)) - abs(P(1,:))) < 1e-12);
assert(abs(abs(det(R1)) - abs(det(P))) < 1e-9*abs(det(P)));

% one output returns Q1
Q1 = qrupdate(eye(2), eye(2), [1; 0], [0; 1]);
assert(isequal(size(Q1), [2 2]));

% errors
threw = false;
try, qrupdate(eye(3), eye(3), [1; 2], [1; 1; 1]); catch, threw = true; end
assert(threw);
threw = false;
try, qrupdate(eye(3), eye(3), [1; 2; 3], [1 1 1]); catch, threw = true; end
assert(threw);
threw = false;
try, qrupdate(eye(3), eye(2), [1; 2; 3], [1; 1]); catch, threw = true; end
assert(threw);

disp('SUCCESS')
