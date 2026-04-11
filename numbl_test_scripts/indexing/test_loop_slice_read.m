% Slice read inside a loop — exercises the loop-JIT path for
% `slice = tensor(:, i)` followed by scalar reads `slice(k)`.
%
% The JIT lowers these as "slice aliases" that substitute through to
% direct scalar reads on the base tensor, avoiding per-iter allocation.
% The cases below probe:
%   - column slice + scalar reads,
%   - row slice from a transposed layout,
%   - multi-dim (3D) column slice,
%   - nested loops with two concurrent slice aliases,
%   - slice alias in a while loop.

% 1) Basic column slice
pts = [ 1 2 3 4 5;
        6 7 8 9 10];
s1 = 0;
s2 = 0;
for i = 1:5
    pt = pts(:, i);
    s1 = s1 + pt(1);
    s2 = s2 + pt(2);
end
assert(s1 == 15, 'col-slice: row 1 sum should be 1+2+3+4+5=15');
assert(s2 == 40, 'col-slice: row 2 sum should be 6+7+8+9+10=40');

% 2) Column slice with arithmetic reads and comparisons — matches
% the stage-5 ptloop pattern.
npts = 20;
nrect = 10;
pts = zeros(2, npts);
for i = 1:npts
    pts(1, i) = mod(i * 3, 11);
    pts(2, i) = mod(i * 7, 13);
end
rects = zeros(4, nrect);
for j = 1:nrect
    rects(1, j) = j - 1;
    rects(2, j) = j + 1;
    rects(3, j) = j - 2;
    rects(4, j) = j + 2;
end
out_pt = zeros(npts * nrect, 1);
out_rect = zeros(npts * nrect, 1);
nhit = 0;
nhit_ref = 0;
for i = 1:npts
    pt = pts(:, i);
    pxi = pt(1);
    pyi = pt(2);
    for j = 1:nrect
        rj = rects(:, j);
        if pxi >= rj(1) && pxi <= rj(2) && pyi >= rj(3) && pyi <= rj(4)
            nhit = nhit + 1;
            out_pt(nhit) = i;
            out_rect(nhit) = j;
        end
    end
end
% Reference: compute directly without slice aliases, for cross-check.
for i = 1:npts
    for j = 1:nrect
        a1 = pts(1, i);
        a2 = pts(2, i);
        b1 = rects(1, j);
        b2 = rects(2, j);
        b3 = rects(3, j);
        b4 = rects(4, j);
        if a1 >= b1 && a1 <= b2 && a2 >= b3 && a2 <= b4
            nhit_ref = nhit_ref + 1;
        end
    end
end
assert(nhit == nhit_ref, 'ptloop: nhit mismatch');

% 3) Slice inside a while loop
w = zeros(4, 5);
for k = 1:5
    for r = 1:4
        w(r, k) = r * 10 + k;
    end
end
k = 0;
total = 0;
while k < 5
    k = k + 1;
    col = w(:, k);
    for r = 1:4
        total = total + col(r);
    end
end
% sum of all entries = sum_{k=1..5, r=1..4} (r*10 + k)
ref = 0;
for k = 1:5
    for r = 1:4
        ref = ref + r * 10 + k;
    end
end
assert(total == ref, 'while-slice: total mismatch');

% 4) Nested loops with two concurrent slice aliases
a = zeros(3, 4);
b = zeros(2, 3);
for j = 1:4
    for i = 1:3
        a(i, j) = i + 100 * j;
    end
end
for j = 1:3
    for i = 1:2
        b(i, j) = i * 1000 + j;
    end
end
aa = 0;
bb = 0;
for j = 1:4
    ca = a(:, j);
    for k = 1:3
        cb = b(:, k);
        aa = aa + ca(1) + ca(2) + ca(3);
        bb = bb + cb(1) + cb(2);
    end
end
ref_a = 0;
ref_b = 0;
for j = 1:4
    for k = 1:3
        ref_a = ref_a + a(1, j) + a(2, j) + a(3, j);
        ref_b = ref_b + b(1, k) + b(2, k);
    end
end
assert(aa == ref_a, 'nested: aa mismatch');
assert(bb == ref_b, 'nested: bb mismatch');

% 5) Slice alias with runtime (non-literal) read index — supported
% because the read-site index is a scalar JitExpr, not required to
% be a compile-time constant.
mat = zeros(3, 6);
for j = 1:6
    for i = 1:3
        mat(i, j) = i + 10 * j;
    end
end
total2 = 0;
for j = 1:6
    col = mat(:, j);
    for r = 1:3
        total2 = total2 + col(r);
    end
end
ref2 = 0;
for j = 1:6
    for r = 1:3
        ref2 = ref2 + mat(r, j);
    end
end
assert(total2 == ref2, 'runtime-idx: mismatch');

% 6) Slice alias in 3D: tensor(:, j, k)
t3 = zeros(2, 3, 4);
for k = 1:4
    for j = 1:3
        for i = 1:2
            t3(i, j, k) = i + 10 * j + 100 * k;
        end
    end
end
t3_sum = 0;
for k = 1:4
    for j = 1:3
        col = t3(:, j, k);
        t3_sum = t3_sum + col(1) + col(2);
    end
end
ref3 = 0;
for k = 1:4
    for j = 1:3
        ref3 = ref3 + t3(1, j, k) + t3(2, j, k);
    end
end
assert(t3_sum == ref3, '3D-slice: mismatch');

disp('SUCCESS');
