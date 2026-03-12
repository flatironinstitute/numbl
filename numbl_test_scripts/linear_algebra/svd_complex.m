% Test complex SVD (requires LAPACK addon)

% Skip if LAPACK addon is not available
try
    svd([1+1i, 0; 0, 1+1i]);
catch e
    if ~isempty(strfind(e.message, 'requires LAPACK'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 1: singular values only for a complex matrix
A = [1+2i, 3+4i; 5+6i, 7+8i];
s = svd(A);
assert(length(s) == 2);
assert(s(1) >= s(2));
assert(s(2) > 0);

% Test 2: full SVD [U, S, V] = svd(A)
[U, S, V] = svd(A);
% Verify A = U * S * V'
R = U * S * V';
assert(norm(R - A) < 1e-10);
% U should be unitary: U'*U = I
assert(norm(U' * U - eye(2)) < 1e-10);
% V should be unitary: V'*V = I
assert(norm(V' * V - eye(2)) < 1e-10);

% Test 3: economy SVD for non-square complex matrix
B = [1+1i, 2+2i, 3+3i; 4+4i, 5+5i, 6+6i];
[U2, S2, V2] = svd(B, 'econ');
% B is 2x3, economy: U2 is 2x2, S2 is 2x2, V2 is 3x2
assert(size(U2, 1) == 2);
assert(size(U2, 2) == 2);
assert(size(S2, 1) == 2);
assert(size(S2, 2) == 2);
assert(size(V2, 1) == 3);
assert(size(V2, 2) == 2);
% Verify reconstruction
R2 = U2 * S2 * V2';
assert(norm(R2 - B) < 1e-10);

% Test 4: economy SVD with numeric 0 flag
[U3, S3, V3] = svd(B, 0);
assert(size(U3, 1) == 2);
assert(size(U3, 2) == 2);
R3 = U3 * S3 * V3';
assert(norm(R3 - B) < 1e-10);

% Test 5: tall complex matrix
C = [1+1i; 2+2i; 3+3i];
[U4, S4, V4] = svd(C);
R4 = U4 * S4 * V4';
assert(norm(R4 - C) < 1e-10);

% Test 6: singular values should always be real and non-negative
s2 = svd(B);
assert(all(s2 >= 0));
assert(isreal(s2));

disp('SUCCESS');
