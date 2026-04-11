% Range-slice write inside a loop — exercises the loop-JIT path for
% `dst(a:b) = src(c:d)` with the same length on both sides. Mirrors the
% chunkie ptloop's grow-and-copy pattern (`out(1:nout) = old(1:nout)`).
%
% Cases below probe:
%   - basic same-length range copy from another tensor
%   - copy interleaved with scalar writes to the same dst
%   - the chunkie growth pattern (reassign dst to a fresh zeros, then copy
%     old data into the prefix, then resume scalar writes)
%   - 2D dst with `[N, 1]` shape (linear range write)
%   - copy where ranges don't start at 1
%   - same-tensor self-copy (overlapping is allowed; subarray clones)

% 1) Basic same-length range copy between two distinct buffers
src = zeros(20, 1);
for i = 1:20
    src(i) = i * 3;
end
dst = zeros(20, 1);
for k = 1:1
    dst(1:20) = src(1:20);
end
for i = 1:20
    assert(dst(i) == i * 3, '1: range copy failed');
end

% 2) Interleaved scalar writes and range copy
a = zeros(10, 1);
b = zeros(10, 1);
for i = 1:10
    b(i) = i * 7;
end
for k = 1:1
    a(1:10) = b(1:10);
    a(5) = 999;
    a(10) = -1;
end
assert(a(1) == 7, '2: copied first element');
assert(a(5) == 999, '2: scalar overwrite of element 5');
assert(a(10) == -1, '2: scalar overwrite of last element');
assert(a(7) == 49, '2: untouched element 7');

% 3) Chunkie growth pattern: reassign dst to a fresh zeros, then range
% copy from the old buffer (now in tmp), then resume scalar writes
% afterwards. The post-reassign hoist refresh is what makes this work.
nout_max = 4;
out_pt = zeros(nout_max, 1);
nhit = 0;
for i = 1:10
    if nhit >= nout_max
        tmp_pt = out_pt;
        nout_max_new = nout_max * 2;
        out_pt = zeros(nout_max_new, 1);
        out_pt(1:nout_max) = tmp_pt(1:nout_max);
        nout_max = nout_max_new;
    end
    nhit = nhit + 1;
    out_pt(nhit) = i * 10;
end
assert(nhit == 10, '3: nhit should be 10');
assert(nout_max == 16, '3: nout_max should have grown 4 -> 8 -> 16');
for i = 1:10
    assert(out_pt(i) == i * 10, sprintf('3: out_pt(%d) wrong', i));
end
% Tail beyond nhit should still be zero
for i = 11:nout_max
    assert(out_pt(i) == 0, sprintf('3: out_pt(%d) should be zero', i));
end

% 4) Linear range write into a 2D [N, 1] tensor
m = zeros(8, 1);
src2 = zeros(8, 1);
for i = 1:8
    src2(i) = 100 + i;
end
for k = 1:1
    m(1:8) = src2(1:8);
end
for i = 1:8
    assert(m(i) == 100 + i, '4: 2D linear range copy failed');
end

% 5) Range copy with a non-1 start
p = zeros(10, 1);
q = zeros(10, 1);
for i = 1:10
    q(i) = i;
end
for k = 1:1
    p(3:7) = q(3:7);
end
assert(p(1) == 0, '5: prefix untouched');
assert(p(2) == 0, '5: prefix untouched');
assert(p(3) == 3, '5: copied');
assert(p(7) == 7, '5: copied');
assert(p(8) == 0, '5: suffix untouched');

% 6) Same-tensor self-copy: shifting a tail forward
% (subarray must clone before write to handle overlap)
s = zeros(10, 1);
for i = 1:10
    s(i) = i;
end
for k = 1:1
    s(1:5) = s(6:10);
end
assert(s(1) == 6, '6: self-copy s(1)');
assert(s(2) == 7, '6: self-copy s(2)');
assert(s(5) == 10, '6: self-copy s(5)');
assert(s(6) == 6, '6: self-copy preserves source range');
assert(s(10) == 10, '6: self-copy preserves source tail');

disp('SUCCESS');
