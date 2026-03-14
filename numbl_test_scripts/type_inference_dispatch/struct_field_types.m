% Test type inference for struct field access

% --- Basic struct field assignment infers Struct type on variable ---
s = struct();
s.x = 5;
s.y = 3.14;
assert(strcmp(__inferred_type_str(s), 'Struct<x: Number, y: Number>'));

% --- Reading a struct field should infer the field's type ---
val = s.x;
assert(strcmp(__inferred_type_str(val), 'Number'));

% --- Struct with mixed field types ---
s2 = struct();
s2.name = 'hello';
s2.value = 42;
s2.data = [1, 2, 3];
assert(strcmp(__inferred_type_str(s2), 'Struct<name: Char, value: Number, data: Tensor<?, real>>'));

% --- Field read propagates type through expressions ---
s3 = struct();
s3.a = 10;
s3.b = 20;
result = s3.a + s3.b;
assert(strcmp(__inferred_type_str(result), 'Number'));

% --- Struct field type used in function specialization ---
s4 = struct();
s4.val = 7;
r = add_one_to_field(s4);
assert(strcmp(__inferred_type_str(r), 'Number'));
assert(r == 8);

disp('SUCCESS')

function result = add_one_to_field(s)
    result = s.val + 1;
end
