% Test chained dot access on class with custom subsref: obj.a.b
% where obj has a subsref that returns a struct for property 'a',
% and 'b' is a field of that struct.
% This tests that the runtime correctly passes a multi-element substruct
% array to subsref and that ind(1) = [] works inside subsref.

obj = ChainedDotHelper();

% Two-level chained access
assert(obj.data.x == 42);
assert(obj.data.y == 99);

% Numeric operations on chained access result
result = 1.2 * obj.data.x;
assert(abs(result - 50.4) < 1e-10);

% Also test struct array construction with cell-valued args
sa = struct('type', {'.', '.'}, 'subs', {'a', 'b'});
assert(numel(sa) == 2);
assert(strcmp(sa(1).type, '.'));
assert(strcmp(sa(1).subs, 'a'));
assert(strcmp(sa(2).subs, 'b'));

% Struct array element deletion
sa(1) = [];
assert(numel(sa) == 1);
assert(strcmp(sa(1).subs, 'b'));

% Delete remaining element
sa(1) = [];
assert(isempty(sa));

disp('SUCCESS');
