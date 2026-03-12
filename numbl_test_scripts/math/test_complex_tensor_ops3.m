% Test cumprod and min/max on complex tensors

% cumprod on complex vector
r1 = cumprod([1+1i, 2+0i, 1-1i]);
% (1+i) = 1+i
% (1+i)*2 = 2+2i
% (2+2i)*(1-i) = 2-2i+2i-2i^2 = 2+2 = 4
assert(abs(r1(1) - (1+1i)) < 1e-10, 'cumprod complex (1)');
assert(abs(r1(2) - (2+2i)) < 1e-10, 'cumprod complex (2)');
assert(abs(r1(3) - 4) < 1e-10, 'cumprod complex (3)');

% min on complex vector — compares by magnitude
A = [3+4i, 1+0i, 0+5i];
[m, idx] = min(A);
assert(abs(m) == 1, 'min complex by abs');
assert(idx == 2, 'min complex index');

% max on complex vector — compares by magnitude
[M, idx2] = max(A);
assert(abs(M) == 5, 'max complex by abs');

% min/max on complex matrix (column-wise)
B = [3+4i 1+0i; 0+1i 2+2i];
r2 = min(B);
% col1: abs(3+4i)=5, abs(0+i)=1 => min = 0+1i
% col2: abs(1)=1, abs(2+2i)=2.83 => min = 1
assert(abs(r2(1) - (0+1i)) < 1e-10, 'min complex matrix col1');
assert(abs(r2(2) - 1) < 1e-10, 'min complex matrix col2');

disp('SUCCESS');
