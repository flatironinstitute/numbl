% Complex operations: prod, sort, std, var, unique, find

% ── prod on complex vector ───────────────────────────────────────
% (1+2i)*(3+4i) = 3+4i+6i+8i^2 = -5+10i
p = prod([1+2i, 3+4i]);
assert(abs(real(p) - (-5)) < 1e-10)
assert(abs(imag(p) - 10) < 1e-10)

% ── prod on complex matrix (along dim 1) ────────────────────────
M = [1+1i 2+0i; 0+1i 3+1i];
% Col 1: (1+1i)*(0+1i) = 0+1i+0i+1i^2 = -1+1i
% Col 2: (2+0i)*(3+1i) = 6+2i
pm = prod(M);
assert(abs(real(pm(1)) - (-1)) < 1e-10)
assert(abs(imag(pm(1)) - 1) < 1e-10)
assert(abs(real(pm(2)) - 6) < 1e-10)
assert(abs(imag(pm(2)) - 2) < 1e-10)

% ── sort on complex vector (by magnitude) ────────────────────────
% MATLAB sorts complex by abs: |3+4i|=5, |1+0i|=1, |0+2i|=2
sv = sort([3+4i, 1+0i, 0+2i]);
assert(abs(real(sv(1)) - 1) < 1e-10)    % |1|=1
assert(abs(real(sv(2)) - 0) < 1e-10)    % |2i|=2
assert(abs(imag(sv(2)) - 2) < 1e-10)
assert(abs(real(sv(3)) - 3) < 1e-10)    % |3+4i|=5
assert(abs(imag(sv(3)) - 4) < 1e-10)

% ── abs on complex vector ────────────────────────────────────────
av = abs([3+4i, 5+12i]);
assert(abs(av(1) - 5) < 1e-10)
assert(abs(av(2) - 13) < 1e-10)

% ── real/imag on complex vectors ─────────────────────────────────
rv = real([1+2i, 3+4i]);
iv = imag([1+2i, 3+4i]);
assert(rv(1) == 1)
assert(rv(2) == 3)
assert(iv(1) == 2)
assert(iv(2) == 4)

% ── complex indexing and assignment ──────────────────────────────
cv = [1+1i, 2+2i, 3+3i];
cv(2) = 5+6i;
assert(abs(real(cv(2)) - 5) < 1e-10)
assert(abs(imag(cv(2)) - 6) < 1e-10)

disp('SUCCESS')
