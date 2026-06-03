% Test conv2 (2-D convolution) builtin

A = [1 2; 3 4];
K = ones(2, 2);

% full convolution (default), size = size(A)+size(B)-1
Cfull = conv2(A, K);
assert(isequal(size(Cfull), [3 3]));
assert(isequal(Cfull, [1 3 2; 4 10 6; 3 7 4]));

% 'same' returns the central part, same size as A
assert(isequal(conv2(A, K, 'same'), [10 6; 7 4]));

% 'valid' returns only the fully-overlapped part
assert(isequal(conv2(A, K, 'valid'), 10));

% identity kernels with 'same'
B = magic(4);
assert(isequal(conv2(B, 1, 'same'), B));
assert(isequal(conv2(B, [0 0 0; 0 1 0; 0 0 0], 'same'), B));

% row vector convolves each row; column vector convolves each column
assert(isequal(conv2([1 2 3; 4 5 6], [1 1]), [1 3 5 3; 4 9 11 6]));
assert(isequal(conv2([1 2 3; 4 5 6], [1; 1]), [1 2 3; 5 7 9; 4 5 6]));

% separable form conv2(u,v,A) == conv2(A, u(:)*v(:).')
P = zeros(10); P(3:7, 3:7) = 1;
u = [1 0 -1]';
v = [1 2 1];
assert(isequal(size(conv2(u, v, P)), [12 12]));
assert(isequal(conv2(u, v, P), conv2(P, u * v)));
assert(isequal(conv2(u, v, P, 'same'), conv2(P, u * v, 'same')));

% complex support
assert(isequal(conv2([1 1i], [1 1]), [1, 1 + 1i, 1i]));

disp('SUCCESS');
