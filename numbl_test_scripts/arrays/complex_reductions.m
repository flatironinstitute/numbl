% Complex tensor reductions: sum, mean, cumsum

cv = [1+2i, 3+4i, 5+6i];

% ── sum on complex row vector → scalar ───────────────────────────
s = sum(cv);
assert(abs(real(s) - 9) < 1e-10)
assert(abs(imag(s) - 12) < 1e-10)

% ── mean on complex row vector → scalar ──────────────────────────
m = mean(cv);
assert(abs(real(m) - 3) < 1e-10)
assert(abs(imag(m) - 4) < 1e-10)

% ── cumsum on complex row vector ─────────────────────────────────
cs = cumsum(cv);
assert(abs(real(cs(1)) - 1) < 1e-10)
assert(abs(imag(cs(1)) - 2) < 1e-10)
assert(abs(real(cs(2)) - 4) < 1e-10)
assert(abs(imag(cs(2)) - 6) < 1e-10)
assert(abs(real(cs(3)) - 9) < 1e-10)
assert(abs(imag(cs(3)) - 12) < 1e-10)

% ── sum on complex matrix along dim 1 ───────────────────────────
M = [1+1i 3+3i; 2+2i 4+4i];
s2 = sum(M);
% Col 1: (1+1i)+(2+2i) = 3+3i, Col 2: (3+3i)+(4+4i) = 7+7i
assert(abs(real(s2(1)) - 3) < 1e-10)
assert(abs(imag(s2(1)) - 3) < 1e-10)
assert(abs(real(s2(2)) - 7) < 1e-10)
assert(abs(imag(s2(2)) - 7) < 1e-10)

% ── sum on complex matrix with explicit dim ──────────────────────
s3 = sum(M, 2);
% Row 1: (1+1i)+(3+3i) = 4+4i, Row 2: (2+2i)+(4+4i) = 6+6i
assert(abs(real(s3(1)) - 4) < 1e-10)
assert(abs(imag(s3(1)) - 4) < 1e-10)
assert(abs(real(s3(2)) - 6) < 1e-10)
assert(abs(imag(s3(2)) - 6) < 1e-10)

% ── mean on complex matrix along dim 1 ──────────────────────────
m2 = mean(M);
assert(abs(real(m2(1)) - 1.5) < 1e-10)
assert(abs(imag(m2(1)) - 1.5) < 1e-10)
assert(abs(real(m2(2)) - 3.5) < 1e-10)
assert(abs(imag(m2(2)) - 3.5) < 1e-10)

% ── cumsum on complex matrix along dim 1 ────────────────────────
cm = cumsum(M);
% Col 1: [1+1i; 3+3i], Col 2: [3+3i; 7+7i]
assert(abs(real(cm(1,1)) - 1) < 1e-10)
assert(abs(imag(cm(1,1)) - 1) < 1e-10)
assert(abs(real(cm(2,1)) - 3) < 1e-10)
assert(abs(imag(cm(2,1)) - 3) < 1e-10)
assert(abs(real(cm(1,2)) - 3) < 1e-10)
assert(abs(imag(cm(1,2)) - 3) < 1e-10)
assert(abs(real(cm(2,2)) - 7) < 1e-10)
assert(abs(imag(cm(2,2)) - 7) < 1e-10)

disp('SUCCESS')
