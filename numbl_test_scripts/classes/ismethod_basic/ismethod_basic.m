% ismethod on classdef objects
c = IsmChild_();
assert(ismethod(c, 'pub'));
assert(ismethod(c, 'stat'));
assert(ismethod(c, 'IsmChild_'));
assert(ismethod(c, 'basepub'));
assert(~ismethod(c, 'prot'));
assert(~ismethod(c, 'basepriv'));
assert(~ismethod(c, 'IsmBase_'));
assert(~ismethod(c, 'v'));
assert(~ismethod(c, 'w'));
assert(~ismethod(c, 'nosuch'));
% handle class methods are inherited
assert(ismethod(c, 'delete'));
assert(ismethod(c, 'isvalid'));
% string name, class-name first argument, object array
assert(ismethod(c, "pub"));
assert(ismethod('IsmChild_', 'pub'));
assert(ismethod([c c], 'pub'));
tf = ismethod(c, 'pub');
assert(islogical(tf) && isscalar(tf));

% @folder class: inline and external method files
f = IsmFolder_();
assert(ismethod(f, 'inline_m'));
assert(ismethod(f, 'extern_m'));
assert(~ismethod(f, 'eq'));

% non-objects
assert(~ismethod(struct('a', 1), 'a'));
assert(~ismethod({}, 'pub'));

disp('SUCCESS');
