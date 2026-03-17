% Test flow-dependent type inference: variable types change at each assignment.

% Straight-line: reassignment changes type
x = 'abc';
assert(strcmp(__inferred_type_str(x), 'Char'));
x = 123;
assert(strcmp(__inferred_type_str(x), 'Number'));
x = [1 2 3];
assert(strcmp(__inferred_type_str(x), 'Tensor<?, real>'));

% Second variable tracks inferred type of RHS expressions
a = 42;
b = a + 1;
assert(strcmp(__inferred_type_str(b), 'Number'));
a = [1 2; 3 4];
b = a * 2;
assert(strcmp(__inferred_type_str(b), 'Tensor<?, real>'));

% After if block, assigned variables reset to Unknown
y = 'hello';
assert(strcmp(__inferred_type_str(y), 'Char'));
if true
    y = 42;
end
assert(strcmp(__inferred_type_str(y), 'Unknown'));

% After for loop, same-type assignment preserves type via join
z = 100;
assert(strcmp(__inferred_type_str(z), 'Number'));
for i = 1:3
    z = z + i;
end
assert(strcmp(__inferred_type_str(z), 'Number'));

% After for loop with type change, join produces Unknown
q = 'text';
for i = 1:2
    q = i;
end
assert(strcmp(__inferred_type_str(q), 'Unknown'));

% Variables not assigned in control flow keep their type
w = 'kept';
if false
    dummy = 1;
end
assert(strcmp(__inferred_type_str(w), 'Char'));

disp('SUCCESS');
