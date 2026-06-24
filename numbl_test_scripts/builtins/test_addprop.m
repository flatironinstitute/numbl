% addprop(obj, name) adds a dynamic property to a dynamicprops handle object.
% Afterwards the property can be read and assigned via obj.name (static or
% dynamic field syntax). addprop returns a meta.DynamicProperty.

o = DynPropObj_();

% addprop returns a meta.DynamicProperty whose Name is the new property.
p = addprop(o, 'extra');
assert(strcmp(p.Name, 'extra'));

% The dynamic property reads as [] before assignment, then holds its value.
assert(isempty(o.extra));
o.extra = [1 2 3];
assert(isequal(o.extra, [1 2 3]));

% Fixed (declared) properties are unaffected.
assert(o.fixed == 1);

% Dynamic field-name assignment to a dynamic property also works.
nm = 'more';
addprop(o, nm);
o.(nm) = 7;
assert(o.(nm) == 7);

disp('SUCCESS')
