% Stage 9 — range-slice write with a whole-tensor source inside a loop.
% Extends test_loop_slice_write.m with the degenerate-RHS shape where
% the source is a plain variable rather than an explicit range slice:
%     dst(a:b) = src;
% The runtime check is `numel(src) == (b - a + 1)`. Mirrors the chunkie
% grow-and-copy line `isp(1:nn) = itemp` where `itemp` is a plain local
% holding the old buffer reference.
%
% `assert_jit_compiled()` is placed inside each outer loop body to
% assert the surrounding loop got JIT-compiled — if the marker call
% survives to the interpreter (because lowering bailed), it throws.

% 1) Basic whole-tensor RHS into a same-length dst range
src = zeros(20, 1);
for i = 1:20
    src(i) = i * 3;
end
dst = zeros(20, 1);
for k = 1:1
    assert_jit_compiled();
    dst(1:20) = src;
end
for i = 1:20
    assert(dst(i) == i * 3, '1: whole-tensor copy failed');
end

% 2) Interleaved with scalar writes — hoisted aliases must stay consistent
a = zeros(10, 1);
b = zeros(10, 1);
for i = 1:10
    b(i) = i * 7;
end
for k = 1:1
    assert_jit_compiled();
    a(1:10) = b;
    a(5) = 999;
    a(10) = -1;
end
assert(a(1) == 7, '2: whole-tensor copied first element');
assert(a(5) == 999, '2: scalar overwrite of element 5');
assert(a(10) == -1, '2: scalar overwrite of last element');
assert(a(7) == 49, '2: untouched element 7');

% 3) Chunkie growth pattern with the stage-9 shape — `tmp_pt` is a
%    plain Var on the RHS (no explicit range). This is the exact shape
%    used by stage_09_slice_write_var_src.m and the target for stage 9.
nout_max = 4;
out_pt = zeros(nout_max, 1);
nhit = 0;
for i = 1:10
    assert_jit_compiled();
    if nhit >= nout_max
        tmp_pt = out_pt;
        nout_max_new = nout_max * 2;
        out_pt = zeros(nout_max_new, 1);
        out_pt(1:nout_max) = tmp_pt;       % whole-tensor RHS
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
for i = 11:nout_max
    assert(out_pt(i) == 0, sprintf('3: out_pt(%d) should be zero', i));
end

% 4) Non-1 starting dst range with whole-tensor source of matching length
p = zeros(10, 1);
q = zeros(5, 1);
for i = 1:5
    q(i) = i + 100;
end
for k = 1:1
    assert_jit_compiled();
    p(3:7) = q;
end
assert(p(1) == 0, '4: prefix untouched');
assert(p(2) == 0, '4: prefix untouched');
assert(p(3) == 101, '4: copied');
assert(p(7) == 105, '4: copied');
assert(p(8) == 0, '4: suffix untouched');

% 5) Length mismatch must throw at runtime (dst range len != numel(src))
mismatch_threw = false;
try
    r1 = zeros(10, 1);
    r2 = zeros(7, 1);
    for k = 1:1
        assert_jit_compiled();
        r1(1:10) = r2;
    end
catch
    mismatch_threw = true;
end
assert(mismatch_threw, '5: length mismatch should have thrown');

disp('SUCCESS');
