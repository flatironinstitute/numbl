% classdef property validation syntax: `Name (dims) Type {validators} = default`.
% numbl's properties-block parser only handled `Name [= default]`, so a bare
% type annotation was consumed as a SECOND property (Value loses its default,
% a bogus `double` property appears), and dims/validators caused a hard syntax
% error that dropped the entire class.

obj = TypedPropClass_();

% type-only annotation must NOT create a phantom property and must keep default
assert(isequal(obj.Value, 7), 'Value (type-only) should default to 7');

% dims + type
assert(isequal(obj.Scale, 2.5), 'Scale ((1,1) double) should default to 2.5');

% validators
assert(isequal(obj.Tag, 9), 'Tag ({validators}) should default to 9');

% plain untyped property still works
assert(strcmp(obj.Plain, 'hello'), 'Plain should default to ''hello''');

fn = fieldnames(obj);
assert(numel(fn) == 4, sprintf('expected 4 properties, got %d', numel(fn)));
assert(~any(strcmp(fn, 'double')), 'type name ''double'' leaked as a property');
assert(all(ismember({'Value', 'Scale', 'Tag', 'Plain'}, fn)), ...
    'property names should be Value/Scale/Tag/Plain');

disp('SUCCESS');
