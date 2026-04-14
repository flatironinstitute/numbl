% Range slice read on RHS inside a loop (stage 21).
% Exercises `r0 = src(a:b)` where src is a real tensor — emits
% `$h.subarrayCopy1r(...)`. Mirrors chunkie `chunk_nearparam.m`
% Newton-iteration pattern `r0 = all0(1:dim)`.

% 1) Basic range read: fixed endpoints
src = zeros(10, 1);
for i = 1:10; src(i) = i * 2; end
total = 0;
for k = 1:5
    r = src(2:5);
    total = total + r(1) + r(4);
end
assert(total == 5 * (4 + 10), '1: fixed range read');

% 2) Range read with dynamic endpoints
dim = 3;
M = 2 * dim;
base = zeros(M, 1);
for i = 1:M; base(i) = 100 + i; end
sum_r0 = 0;
sum_r1 = 0;
for k = 1:20
    r0 = base(1:dim);
    r1 = base(dim+1:M);
    sum_r0 = sum_r0 + r0(1) + r0(dim);
    sum_r1 = sum_r1 + r1(1) + r1(dim);
end
% r0 covers base(1..3) = 101..103; r0(1)+r0(3) = 101 + 103 = 204
% r1 covers base(4..6) = 104..106; r1(1)+r1(3) = 104 + 106 = 210
assert(sum_r0 == 204 * 20, '2: r0 sum');
assert(sum_r1 == 210 * 20, '2: r1 sum');

% 3) Range length 1 (degenerate)
s = zeros(5, 1);
for i = 1:5; s(i) = i; end
total3 = 0;
for k = 1:10
    t = s(3:3);
    total3 = total3 + t(1);
end
assert(total3 == 30, '3: length-1 range');

% 4) Range read in a branch — lazy eval per iter
src4 = zeros(8, 1);
for i = 1:8; src4(i) = i; end
total4 = 0;
for i = 1:8
    if mod(i, 2) == 0
        r = src4(1:i);
        total4 = total4 + r(i);
    end
end
% i = 2,4,6,8: r(i) = src4(i) = i. Sum = 2+4+6+8 = 20.
assert(total4 == 20, '4: range read in branch');

% 5) Range that reaches 'end' — src(k:end) — stage 21 scope
% (Only covered if `end` lowers; in chunkie's `all0(2*dim+1:end)` the
% parser emits `end` as a special node. This test uses a numeric
% endpoint to stay within stage 21's narrow shape.)
src5 = zeros(10, 1);
for i = 1:10; src5(i) = i * 3; end
total5 = 0;
N = 10;
for k = 1:3
    r = src5(5:N);
    total5 = total5 + r(1) + r(end);
end
% r = src5(5:10) = [15, 18, 21, 24, 27, 30]; r(1)=15, r(end)=30; sum = 45 * 3
assert(total5 == 45 * 3, '5: range to end-as-variable');

% 6) Out-of-bounds start should throw
ok = false;
oob = zeros(5, 1);
try
    for k = 1:1
        r = oob(0:3);
    end
catch
    ok = true;
end
assert(ok, '6: out-of-bounds start');

disp('SUCCESS');
