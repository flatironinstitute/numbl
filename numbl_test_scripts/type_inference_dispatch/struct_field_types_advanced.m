% Advanced struct field type inference tests — tricky cases and known limitations

% =====================================================================
% PART 1: Cases that should work
% =====================================================================

% --- Field reassigned to same type ---
s = struct();
s.x = 1;
s.x = 2;
assert(strcmp(__inferred_type_str(s), 'Struct<x: Number>'));
assert(strcmp(__inferred_type_str(s.x), 'Number'));

% --- Field reassigned to different type degrades to Unknown ---
s2 = struct();
s2.x = 5;
s2.x = 'hello';
% Unification of Number and Char → Unknown
assert(strcmp(__inferred_type_str(s2.x), 'Unknown'));

% --- Direct field assignment without struct() first ---
clear s3;
s3.a = 10;
s3.b = [1 2 3];
assert(strcmp(__inferred_type_str(s3), 'Struct<a: Number, b: Tensor<?, real>>'));
assert(strcmp(__inferred_type_str(s3.a), 'Number'));
assert(strcmp(__inferred_type_str(s3.b), 'Tensor<?, real>'));

% --- Struct field holding a complex number ---
s4 = struct();
s4.z = 3 + 4i;
assert(strcmp(__inferred_type_str(s4.z), 'ComplexNumber'));

% --- Struct field holding a logical ---
s5 = struct();
s5.flag = true;
assert(strcmp(__inferred_type_str(s5.flag), 'Boolean'));

% --- Struct field holding a cell ---
s6 = struct();
s6.items = {1, 'two', 3};
assert(strcmp(__inferred_type_str(s6), 'Struct<items: Cell<elementType=?, length=?>>'));

% --- Reading unknown field returns Unknown ---
s7 = struct();
s7.x = 1;
assert(strcmp(__inferred_type_str(s7.y), 'Unknown'));

% --- Arithmetic on struct fields ---
s8 = struct();
s8.a = 10;
s8.b = 20;
result = s8.a * s8.b + 1;
assert(strcmp(__inferred_type_str(result), 'Number'));
assert(result == 201);

% --- Struct field type propagates through function ---
s9 = struct();
s9.val = 42;
r = double_field(s9);
assert(strcmp(__inferred_type_str(r), 'Number'));
assert(r == 84);

% =====================================================================
% PART 2: Weakness probes — currently limited but correct
% =====================================================================

% --- struct() builtin with field-value pairs: field types not tracked ---
% The builtin returns Struct<> (empty) because the check function
% doesn't analyze arguments. Field reads after this are Unknown.
s10 = struct('x', 1, 'y', 2);
% Current: Struct<> (no field info from builtin args)
disp(['struct with args type: ' __inferred_type_str(s10)]);
disp(['struct with args field: ' __inferred_type_str(s10.x)]);

% --- Nested struct assignment: only top-level field tracked ---
s11 = struct();
s11.inner = struct();
s11.inner.val = 99;
disp(['nested struct base type: ' __inferred_type_str(s11)]);
disp(['nested struct inner type: ' __inferred_type_str(s11.inner)]);
disp(['nested struct inner.val type: ' __inferred_type_str(s11.inner.val)]);

% --- Dynamic field assignment: not tracked ---
s12 = struct();
fname = 'x';
s12.(fname) = 5;
disp(['dynamic field assign type: ' __inferred_type_str(s12)]);
disp(['dynamic field read type: ' __inferred_type_str(s12.(fname))]);

% --- Struct from function return then field assignment ---
s13 = make_struct();
s13.extra = 'added';
disp(['func return struct type: ' __inferred_type_str(s13)]);

disp('SUCCESS')

function result = double_field(s)
    result = s.val * 2;
end

function s = make_struct()
    s = struct();
    s.x = 1;
    s.y = 2;
end
