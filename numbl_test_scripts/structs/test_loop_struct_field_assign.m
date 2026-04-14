% Struct field assignment lvalue inside a JIT loop (stage 22).
% Exercises the AssignMember IR path + structUnshare_h COW for struct
% params. Mirrors chunkie adapgausskerneval's srcinfo/targinfo struct
% construction inside oneintp.

% 1) Basic `s = struct(); s.f = v;` inside a loop
total = 0;
for i = 1:100
    s = struct();
    s.a = i;
    s.b = i * 2;
    total = total + s.a + s.b;
end
assert(total == 100 * 101 / 2 + 2 * (100 * 101 / 2), '1: basic struct field assign');

% 2) `s = []; s.f = v;` (MATLAB idiom — empty matrix promoted to struct)
total2 = 0;
for i = 1:100
    s = [];
    s.x = i * 3;
    s.y = i - 5;
    total2 = total2 + s.x + s.y;
end
assert(total2 == 3 * (100 * 101 / 2) + (100 * 101 / 2 - 5 * 100), ...
    '2: empty-to-struct promotion');

% 3) Tensor-valued field
tot3 = 0;
for i = 1:50
    s = struct();
    s.vec = [i; i * 2; i * 3];
    tot3 = tot3 + s.vec(1) + s.vec(2) + s.vec(3);
end
assert(tot3 == 6 * (50 * 51 / 2), '3: tensor field');

% 4) Multiple fields + re-assignment
tot4 = 0;
for i = 1:30
    s = struct();
    s.a = i;
    s.b = i + 10;
    s.a = s.a + s.b;        % re-assign a; read b + a
    tot4 = tot4 + s.a;
end
% sum_{i=1}^{30} (i + i+10) = 2*(30*31/2) + 300 = 930 + 300 = 1230
assert(tot4 == 1230, '4: multiple/reassign fields');

% 5) Value semantics: callee mutating a param struct must not affect caller
s_outer = struct('x', 5);
total5 = 0;
for j = 1:10
    bump_struct(s_outer);
    total5 = total5 + s_outer.x;
end
assert(total5 == 5 * 10, '5: callee cannot mutate caller struct');

disp('SUCCESS');

function bump_struct(s)
    s.x = 999;
end
