% Regression: a class method declared with an explicit empty output list
% `function [] = name(...)` must parse and run (the ClassParser previously
% rejected the empty `[]`).

obj = EmptyOutMethod_();
obj.doThing(7);
assert(obj.Value == 8, 'empty-output method ran and mutated the handle');

disp('SUCCESS');
