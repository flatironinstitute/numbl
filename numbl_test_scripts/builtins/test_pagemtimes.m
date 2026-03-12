% Test pagemtimes - page-wise matrix multiplication

% Basic 3-D case
A = reshape(1:12, [2,2,3]);
B = reshape(13:24, [2,2,3]);
C = pagemtimes(A, B);
assert(isequal(size(C), [2,2,3]));
% Page 1: A(:,:,1) = [1 3; 2 4], B(:,:,1) = [13 15; 14 16]
% [1 3; 2 4] * [13 15; 14 16] = [55 63; 82 94]
assert(C(1,1,1) == 55);
assert(C(2,1,1) == 82);
assert(C(1,2,1) == 63);
assert(C(2,2,1) == 94);

% Matrix * N-D array (broadcast matrix across pages)
M = [1 0; 0 2];
D = pagemtimes(M, A);
assert(isequal(size(D), [2,2,3]));
% Page 1: M * [1 3; 2 4] = [1 3; 4 8]
assert(D(1,1,1) == 1);
assert(D(2,1,1) == 4);
assert(D(1,2,1) == 3);
assert(D(2,2,1) == 8);

% Non-square pages
X = reshape(1:12, [2,3,2]);
Y = reshape(1:6, [3,1,2]);
Z = pagemtimes(X, Y);
assert(isequal(size(Z), [2,1,2]));

% Transpose options
A2 = reshape(1:8, [2,2,2]);
B2 = reshape(9:16, [2,2,2]);
CT = pagemtimes(A2, 'transpose', B2, 'none');
% Page 1: A2(:,:,1)' * B2(:,:,1) = [1 2; 3 4]' * [9 11; 10 12]
%        = [1 2; 3 4] * [9 11; 10 12] = [29 35; 67 81]
assert(CT(1,1,1) == 29);

fprintf('SUCCESS\n');
