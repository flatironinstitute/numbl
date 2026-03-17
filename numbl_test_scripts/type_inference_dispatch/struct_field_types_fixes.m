% Tests for struct field type inference fixes
% Each section targets a specific weakness

% =====================================================================
% FIX 1: struct('x', 1, 'y', 2) should infer field types from args
% =====================================================================

% --- struct() with field-value pairs ---
s1 = struct('x', 1, 'y', 2);
assert(strcmp(__inferred_type_str(s1), 'Struct<x: Number, y: Number>'));
assert(strcmp(__inferred_type_str(s1.x), 'Number'));
assert(strcmp(__inferred_type_str(s1.y), 'Number'));

% --- struct() with mixed types ---
s1b = struct('name', 'alice', 'age', 30, 'scores', [90, 85, 92]);
assert(strcmp(__inferred_type_str(s1b), 'Struct<name: Char, age: Number, scores: Tensor<?, real>>'));
assert(strcmp(__inferred_type_str(s1b.name), 'Char'));
assert(strcmp(__inferred_type_str(s1b.age), 'Number'));
assert(strcmp(__inferred_type_str(s1b.scores), 'Tensor<?, real>'));

% --- struct() with no args still works ---
s1c = struct();
assert(strcmp(__inferred_type_str(s1c), 'Struct<>'));

% --- struct() with field-value, then additional field assignment ---
s1d = struct('x', 1);
s1d.y = 'hello';
assert(strcmp(__inferred_type_str(s1d), 'Struct<x: Number, y: Char>'));

% --- Values should be correct at runtime ---
assert(s1.x == 1);
assert(s1.y == 2);
assert(strcmp(s1b.name, 'alice'));

% =====================================================================
% FIX 2: Nested struct assignment (s.a.b = 5) tracks through chain
% =====================================================================

% --- Two-level nesting ---
s2 = struct();
s2.inner = struct();
s2.inner.val = 99;
assert(strcmp(__inferred_type_str(s2.inner), 'Struct<val: Number>'));
assert(strcmp(__inferred_type_str(s2.inner.val), 'Number'));
assert(s2.inner.val == 99);

% --- Three-level nesting ---
s2b = struct();
s2b.a = struct();
s2b.a.b = struct();
s2b.a.b.c = 42;
assert(strcmp(__inferred_type_str(s2b.a.b.c), 'Number'));
assert(s2b.a.b.c == 42);

% --- Nested struct with multiple fields at each level ---
s2c = struct();
s2c.pos = struct();
s2c.pos.x = 1;
s2c.pos.y = 2;
s2c.name = 'point';
assert(strcmp(__inferred_type_str(s2c.pos.x), 'Number'));
assert(strcmp(__inferred_type_str(s2c.pos.y), 'Number'));
assert(strcmp(__inferred_type_str(s2c.name), 'Char'));

% --- Nested struct built without explicit struct() ---
clear s2d;
s2d.config.debug = true;
s2d.config.level = 3;
assert(strcmp(__inferred_type_str(s2d.config.debug), 'Boolean'));
assert(strcmp(__inferred_type_str(s2d.config.level), 'Number'));

% =====================================================================
% FIX 3: Dynamic field assignment marks variable as Struct
% =====================================================================

% --- Dynamic field on fresh variable ---
s3 = struct();
fname = 'x';
s3.(fname) = 5;
% We can't know the field name statically, but s3 should still be Struct
assert(strcmp(__inferred_type_str(s3), 'Struct<>'));

% --- Dynamic field doesn't clobber existing field info ---
s3b = struct();
s3b.known = 10;
fname2 = 'dynamic';
s3b.(fname2) = 20;
% s3b should still know about .known
assert(strcmp(__inferred_type_str(s3b), 'Struct<known: Number>'));
assert(strcmp(__inferred_type_str(s3b.known), 'Number'));

% =====================================================================
% Whole-struct reassignment: flow-dependent analysis resets fields
% =====================================================================

% Type inference is flow-dependent: reassignment replaces the type at
% this program point, so only fields added after the last reassignment
% are tracked.
s4 = struct();
s4.x = 1;
s4 = struct();
s4.y = 2;
% Only y is tracked because the second struct() assignment resets the type
assert(strcmp(__inferred_type_str(s4), 'Struct<y: Number>'));

disp('SUCCESS')
