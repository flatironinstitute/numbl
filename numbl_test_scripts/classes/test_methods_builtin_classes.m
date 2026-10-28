% methods / ismethod on built-in classes without a classdef (MException,
% graph, dictionary). numbl lists the subset of MATLAB's methods it
% implements, so check membership rather than the full list.
ME = MException('a:b', 'hi');
m = methods(ME);
assert(iscell(m) && size(m, 2) == 1);
for name = {'MException', 'addCause', 'addCorrection', 'getReport', ...
            'isequal', 'rethrow', 'throw', 'throwAsCaller'}
    assert(any(strcmp(m, name{1})), name{1});
end
assert(isequal(sort(m), m));
assert(isequal(methods('MException'), m));
assert(ismethod(ME, 'getReport'));
assert(~ismethod(ME, 'nosuchmethod'));

G = graph([1 2], [2 3]);
m = methods(G);
for name = {'addedge', 'conncomp', 'degree', 'graph', 'laplacian', ...
            'numedges', 'numnodes'}
    assert(any(strcmp(m, name{1})), name{1});
end
assert(ismethod(G, 'degree'));

D = dictionary(1, 2);
m = methods(D);
for name = {'dictionary', 'entries', 'insert', 'isConfigured', 'isKey', ...
            'keys', 'lookup', 'numEntries', 'remove', 'types', 'values'}
    assert(any(strcmp(m, name{1})), name{1});
end
assert(ismethod(D, 'isKey'));

disp('SUCCESS');
