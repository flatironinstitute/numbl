% Test that V(k).field = rhs inside a class method preserves class type.
% In MATLAB classdef, methods have privileged access to their own instances,
% bypassing overloaded subsref/subsasgn for built-in indexing.

obj = MyIndexedObj(10);
disp(class(obj));
obj = fixup(obj);
disp(class(obj));

if ~strcmp(class(obj), 'MyIndexedObj')
    error('Class type lost in method using V(k).field = rhs');
end
if obj.data ~= 11
    error('Method fixup did not increment data correctly');
end

disp('SUCCESS');
