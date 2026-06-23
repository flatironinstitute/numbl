% triangulation / TriRep: freeBoundary and vertexAttachments queries.
% Verified against MATLAB R2025b (TriRep is deprecated there but still works).

% A fan of 4 triangles around a center vertex (1), boundary 2-3-4-5-2.
f = [1 2 3; 1 3 4; 1 4 5; 1 5 2];
v = [0 0; 1 0; 1 1; 0 1; -1 0];

TR = triangulation(f, v);

% freeBoundary: the 4 outer edges, as a connected loop of distinct vertices.
B = freeBoundary(TR);
assert(size(B, 1) == 4, 'freeBoundary returned wrong number of edges');
assert(isequal(sort(unique(B(:))), [2;3;4;5]), 'freeBoundary vertices wrong');
% Consecutive edges must connect head-to-tail (a closed loop).
assert(isequal(B(:,2), B([2:end,1],1)), 'freeBoundary is not a connected loop');

% vertexAttachments: center vertex touches all 4 triangles.
A = vertexAttachments(TR);
assert(iscell(A) && numel(A) == size(v,1), 'vertexAttachments wrong shape');
assert(isequal(sort(A{1}), [1 2 3 4]), 'center vertex attachments wrong');
assert(numel(A{3}) == 2, 'corner vertex should touch 2 triangles');

% TriRep (legacy) inherits the same methods and exposes X / Triangulation.
TRr = TriRep(f, v);
assert(isequal(freeBoundary(TRr), B), 'TriRep.freeBoundary disagrees');
assert(isequal(TRr.X, v), 'TriRep.X wrong');
assert(isequal(TRr.Triangulation, f), 'TriRep.Triangulation wrong');

disp('SUCCESS')
