% Test that class methods take priority over constructors with the same name.
% When calling InfoClass(hostObj, 'get') where hostObj is a HostClass instance,
% MATLAB dispatches to @HostClass/InfoClass (the method), not @InfoClass/InfoClass
% (the constructor).

h = HostClass([10, 20, 30]);

% This should call h.InfoClass('get'), not InfoClass(h, 'get') as constructor
result = InfoClass(h, 'get');

assert(isequal(result, [10, 20, 30]));

disp('SUCCESS');
