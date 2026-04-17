% Stage 11 — empty matrix literal + vertical concat growth inside a
% JIT loop. Mirrors the chunkie `flagnear_rectangle` "found list"
% pattern:
%     it = [];
%     for jj = ...
%         if in
%             it = [it; i];
%         end
%     end
%
% The lowering recognizes `[base; value]` where `base` is a real
% column-vector (or empty) tensor and `value` is a numeric scalar, and
% emits `$h.vconcatGrow1r(base, value)` — a per-iter allocate-and-copy
% helper that returns a fresh `(k+1, 1)` tensor.
%
% ``%!numbl:assert_jit`` is placed inside each outer loop body to
% assert the surrounding loop got JIT-compiled — if the marker call
% survives to the interpreter (because lowering bailed), it throws.

% 1) Basic growth: build a column vector 1..n via `[it; i]`.
it = [];
for i = 1:20
    %!numbl:assert_jit
    it = [it; i];
end
assert(length(it) == 20, '1: length after grow');
assert(it(1) == 1, '1: first element');
assert(it(20) == 20, '1: last element');
for i = 1:20
    assert(it(i) == i, sprintf('1: it(%d) wrong', i));
end

% 2) Conditional growth: append only on even values. Verify count and
%    sum against a separate scalar accumulator to check correctness.
it2 = [];
expected_count = 0;
expected_sum = 0;
for i = 1:50
    %!numbl:assert_jit
    if mod(i, 2) == 0
        it2 = [it2; i];
        expected_count = expected_count + 1;
        expected_sum = expected_sum + i;
    end
end
assert(length(it2) == expected_count, '2: even count');
actual_sum = 0;
for k = 1:length(it2)
    actual_sum = actual_sum + it2(k);
end
assert(actual_sum == expected_sum, '2: even sum');

% 3) Reset to `[]` each outer iter (mirrors chunkie's per-leaf reset).
%    Inner loop may or may not append. `isempty(it)` must work before
%    and after the inner loop.
totallen = 0;
totalsum = 0;
for i = 1:30
    %!numbl:assert_jit
    it3 = [];
    for j = 1:5
        if mod(i + j, 3) == 0
            it3 = [it3; i * 10 + j];
        end
    end
    if ~isempty(it3)
        totallen = totallen + length(it3);
        totalsum = totalsum + it3(1);
    end
end
% Recompute the reference totals with a scalar-only inner loop.
ref_totallen = 0;
ref_totalsum = 0;
for i = 1:30
    first_hit = 0;
    first_seen = false;
    hit_count = 0;
    for j = 1:5
        if mod(i + j, 3) == 0
            hit_count = hit_count + 1;
            if ~first_seen
                first_hit = i * 10 + j;
                first_seen = true;
            end
        end
    end
    if hit_count > 0
        ref_totallen = ref_totallen + hit_count;
        ref_totalsum = ref_totalsum + first_hit;
    end
end
assert(totallen == ref_totallen, '3: totallen mismatch');
assert(totalsum == ref_totalsum, '3: totalsum mismatch');

% 4) Grow across outer iters and consume after the loop. Verifies that
%    the tensor's final shape flows out of the JIT'd loop correctly.
it4 = [];
for i = 1:7
    %!numbl:assert_jit
    it4 = [it4; i * i];
end
assert(length(it4) == 7, '4: post-loop length');
assert(it4(3) == 9, '4: it4(3)');
assert(it4(7) == 49, '4: it4(7)');

% 5) Scalar read after an empty-then-nonempty transition inside the
%    same outer iter — exercises the hoist refresh on both assigns.
acc = 0;
for i = 1:10
    %!numbl:assert_jit
    it5 = [];
    it5 = [it5; i];
    it5 = [it5; i * 2];
    acc = acc + it5(1) + it5(2);
end
% it5(1) = i, it5(2) = 2i, so acc = sum(i + 2i for i=1..10) = 3 * 55 = 165
assert(acc == 165, '5: acc');

disp('SUCCESS');
