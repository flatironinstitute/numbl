% Test struct field type inference with control flow and edge cases

% --- Struct field assigned in both branches (same type) ---
s = struct();
if true
    s.x = 1;
else
    s.x = 2;
end
% Both branches assign Number to s.x
assert(strcmp(__inferred_type_str(s), 'Struct<x: Number>'));
assert(strcmp(__inferred_type_str(s.x), 'Number'));

% --- Struct field assigned in only one branch ---
s2 = struct();
s2.a = 1;
if true
    s2.b = 2;
end
% s2 should have both fields since both assignments are unconditionally lowered
assert(strcmp(__inferred_type_str(s2), 'Struct<a: Number, b: Number>'));

% --- Struct built up in a loop ---
s3 = struct();
s3.total = 0;
for i = 1:5
    s3.total = s3.total + i;
end
assert(strcmp(__inferred_type_str(s3.total), 'Number'));
assert(s3.total == 15);

% --- Struct assigned then reassigned as whole struct ---
s4 = struct();
s4.x = 1;
s4 = struct();
s4.y = 2;
% After reassignment, type is unified: Struct<x: Number> + Struct<> + Struct<y: Number>
disp(['reassigned struct type: ' __inferred_type_str(s4)]);

% --- Multiple structs, fields don't bleed ---
a = struct();
a.x = 1;
b = struct();
b.y = 'hello';
assert(strcmp(__inferred_type_str(a), 'Struct<x: Number>'));
assert(strcmp(__inferred_type_str(b), 'Struct<y: Char>'));
assert(strcmp(__inferred_type_str(a.x), 'Number'));
assert(strcmp(__inferred_type_str(b.y), 'Char'));

% --- Struct with tensor field used in computation ---
s5 = struct();
s5.data = [1, 2, 3, 4, 5];
s5.mean_val = sum(s5.data) / 5;
assert(strcmp(__inferred_type_str(s5.data), 'Tensor<?, real>'));
assert(abs(s5.mean_val - 3) < 1e-10);

% --- Struct comparison: field types determine expression type ---
s6 = struct();
s6.val = 42;
flag = s6.val > 10;
assert(strcmp(__inferred_type_str(flag), 'Boolean'));
assert(flag);

disp('SUCCESS')
