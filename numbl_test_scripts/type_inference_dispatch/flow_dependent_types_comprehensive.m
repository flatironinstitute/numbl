% Comprehensive tests for flow-dependent type inference.
% Tests edge cases for how variable types change across statements
% and through control flow join points.

% =====================================================================
% 1. Straight-line reassignment: type updates at each assignment
% =====================================================================

v = 1;
assert(strcmp(__inferred_type_str(v), 'Number'));
v = 'hello';
assert(strcmp(__inferred_type_str(v), 'Char'));
v = [1 2 3];
assert(strcmp(__inferred_type_str(v), 'Tensor<?, real>'));
v = true;
assert(strcmp(__inferred_type_str(v), 'Boolean'));
v = 42;
assert(strcmp(__inferred_type_str(v), 'Number'));

% =====================================================================
% 2. Downstream expressions see the current type
% =====================================================================

x = 10;
y = x + 5;
assert(strcmp(__inferred_type_str(y), 'Number'));
x = [1 2; 3 4];
y = x + 1;
assert(strcmp(__inferred_type_str(y), 'Tensor<?, real>'));

% =====================================================================
% 3. If/else — same type in all branches preserves type
% =====================================================================

r = 0;
if true
    r = 1;
else
    r = 2;
end
assert(strcmp(__inferred_type_str(r), 'Number'));

% =====================================================================
% 4. If/else — different types in branches → Unknown via unify
% =====================================================================

m = 0;
if true
    m = 'text';
else
    m = 42;
end
assert(strcmp(__inferred_type_str(m), 'Unknown'));

% =====================================================================
% 5. If without else — pre-branch type included in join
% =====================================================================

% Same type: Number before, Number in branch → Number
n1 = 10;
if false
    n1 = 20;
end
assert(strcmp(__inferred_type_str(n1), 'Number'));

% Different type: Char before, Number in branch → Unknown
n2 = 'abc';
if false
    n2 = 42;
end
assert(strcmp(__inferred_type_str(n2), 'Unknown'));

% =====================================================================
% 6. Elseif chains
% =====================================================================

c = 0;
if false
    c = 1;
elseif false
    c = 2;
elseif true
    c = 3;
end
% All branches assign Number, pre-branch is Number → Number
assert(strcmp(__inferred_type_str(c), 'Number'));

% Mixed types across elseif
c2 = 0;
if false
    c2 = 'a';
elseif true
    c2 = 42;
else
    c2 = [1 2];
end
assert(strcmp(__inferred_type_str(c2), 'Unknown'));

% =====================================================================
% 7. Nested if inside if
% =====================================================================

p = 100;
if true
    if true
        p = 200;
    end
end
% Outer if: no else, pre-branch (100=Number) joined with post-then.
% Inner if: no else, pre-branch (100=Number) joined with post-then (200=Number).
% Inner join → Number. Outer join → Number.
assert(strcmp(__inferred_type_str(p), 'Number'));

% Nested with type change
p2 = 100;
if true
    if true
        p2 = 'changed';
    end
end
assert(strcmp(__inferred_type_str(p2), 'Unknown'));

% =====================================================================
% 8. For loop — same type preserved
% =====================================================================

total = 0;
for k = 1:10
    total = total + k;
end
assert(strcmp(__inferred_type_str(total), 'Number'));
assert(total == 55);

% =====================================================================
% 9. For loop — type change produces Unknown
% =====================================================================

val = 'start';
for k = 1:3
    val = k * 2;
end
assert(strcmp(__inferred_type_str(val), 'Unknown'));

% =====================================================================
% 10. While loop — same type preserved
% =====================================================================

count = 0;
idx = 1;
while idx <= 5
    count = count + 1;
    idx = idx + 1;
end
assert(strcmp(__inferred_type_str(count), 'Number'));
assert(count == 5);

% =====================================================================
% 11. While loop — type change
% =====================================================================

wval = 'init';
wi = 0;
while wi < 1
    wval = 42;
    wi = wi + 1;
end
assert(strcmp(__inferred_type_str(wval), 'Unknown'));

% =====================================================================
% 12. Switch/case — same type all cases
% =====================================================================

sv = 0;
switch 2
    case 1
        sv = 10;
    case 2
        sv = 20;
    otherwise
        sv = 30;
end
assert(strcmp(__inferred_type_str(sv), 'Number'));

% =====================================================================
% 13. Switch/case — mixed types
% =====================================================================

sv2 = 0;
switch 1
    case 1
        sv2 = 'hello';
    otherwise
        sv2 = 42;
end
assert(strcmp(__inferred_type_str(sv2), 'Unknown'));

% =====================================================================
% 14. Switch without otherwise — pre-branch included
% =====================================================================

sv3 = 'default';
switch 99
    case 1
        sv3 = 42;
end
% No otherwise → pre-branch 'default' (Char) joins with case (Number) → Unknown
assert(strcmp(__inferred_type_str(sv3), 'Unknown'));

% Same type everywhere
sv4 = 10;
switch 99
    case 1
        sv4 = 20;
end
assert(strcmp(__inferred_type_str(sv4), 'Number'));

% =====================================================================
% 15. Try/catch
% =====================================================================

tv = 0;
try
    tv = 42;
catch
    tv = 99;
end
assert(strcmp(__inferred_type_str(tv), 'Number'));

% Try/catch with type change
tv2 = 0;
try
    tv2 = 'oops';
catch
    tv2 = 42;
end
assert(strcmp(__inferred_type_str(tv2), 'Unknown'));

% =====================================================================
% 16. Variable assigned after control flow gets precise type again
% =====================================================================

u = 'text';
if true
    u = 42;
end
% u is Unknown here
assert(strcmp(__inferred_type_str(u), 'Unknown'));
u = [1 2 3];
% After reassignment, type is precise again
assert(strcmp(__inferred_type_str(u), 'Tensor<?, real>'));

% =====================================================================
% 17. Unrelated variables unaffected by control flow
% =====================================================================

stable = 'constant';
changing = 0;
if true
    changing = 'modified';
end
assert(strcmp(__inferred_type_str(stable), 'Char'));
assert(strcmp(__inferred_type_str(changing), 'Unknown'));

% =====================================================================
% 18. Multiple variables assigned in same control flow
% =====================================================================

a1 = 0;
b1 = 'x';
if true
    a1 = 10;
    b1 = 20;
else
    a1 = 30;
    b1 = 'y';
end
% a1: Number in both branches → Number
% b1: Number in then, Char in else → Unknown
assert(strcmp(__inferred_type_str(a1), 'Number'));
assert(strcmp(__inferred_type_str(b1), 'Unknown'));

% =====================================================================
% 19. For loop variable type
% =====================================================================

for ii = 1:5
    assert(strcmp(__inferred_type_str(ii), 'Number'));
end

% =====================================================================
% 20. Struct member assign flow — fields accumulate, reassignment resets
% =====================================================================

s = struct();
s.a = 1;
assert(strcmp(__inferred_type_str(s), 'Struct<a: Number>'));
s.b = 'hello';
assert(strcmp(__inferred_type_str(s), 'Struct<a: Number, b: Char>'));
% Whole-struct reassignment resets
s = struct();
assert(strcmp(__inferred_type_str(s), 'Struct<>'));
s.c = [1 2 3];
assert(strcmp(__inferred_type_str(s), 'Struct<c: Tensor<?, real>>'));

% =====================================================================
% 21. Control flow with only one branch assigning a NEW variable
% =====================================================================

if true
    new_var = 42;
end
% new_var was not defined before the if, assigned only in then-body
% After if: join with pre-branch (undefined) → result depends on branch
assert(strcmp(__inferred_type_str(new_var), 'Number'));

% =====================================================================
% 22. If/else where both branches assign same class type
% =====================================================================

% (Using struct as proxy since class tests need --add-script-path)
s1 = struct();
if true
    s1.val = 1;
else
    s1.val = 2;
end
assert(strcmp(__inferred_type_str(s1), 'Struct<val: Number>'));

% =====================================================================
% 23. Deeply nested control flow
% =====================================================================

deep = 0;
for i = 1:2
    if true
        for j = 1:2
            deep = deep + 1;
        end
    end
end
assert(strcmp(__inferred_type_str(deep), 'Number'));
assert(deep == 4);

% =====================================================================
% 24. Boolean result from comparison after type change
% =====================================================================

cmp = 10;
assert(strcmp(__inferred_type_str(cmp > 5), 'Boolean'));
cmp = [1 2 3];
assert(strcmp(__inferred_type_str(cmp > 2), 'Tensor<?, real, logical>'));

disp('SUCCESS');
