% Growing an N-D array via colon subscripts from an empty/undefined base.
% A trailing colon dimension (beyond the empty base's rank) must take its
% size from the RHS, e.g. a(1,:,:) = <3x4> on an empty `a` -> size [1 3 4].
% (This is the pattern @msh_cartesian/msh_evaluate_col uses to fill
% quad_nodes(idim,:,:).)

% From an empty literal
a = [];
a(1,:,:) = reshape(1:12, 3, 4);
assert(isequal(size(a), [1 3 4]));
assert(a(1,2,3) == 8);
assert(a(1,3,4) == 12);

% From an undefined variable
clear b
b(1,:,:) = reshape(1:12, 3, 4);
assert(isequal(size(b), [1 3 4]));

% From an undefined struct field
clear s
s.quad(1,:,:) = reshape(1:12, 3, 4);
assert(isequal(size(s.quad), [1 3 4]));
assert(s.quad(1,2,3) == 8);

% Second row appended into the same 3-D array
a(2,:,:) = reshape(13:24, 3, 4);
assert(isequal(size(a), [2 3 4]));
assert(a(2,1,1) == 13);
assert(a(1,2,3) == 8);   % first slice preserved

% Existing (non-empty) array slice assignment still works
c = zeros(2,3,4);
c(1,:,:) = reshape(1:12, 3, 4);
assert(isequal(size(c), [2 3 4]));
assert(c(1,3,4) == 12);

% A known-but-empty leading dim keeps the declared trailing size
d = zeros(0,3);
d(1,:) = [7 8 9];
assert(isequal(size(d), [1 3]));
assert(isequal(d, [7 8 9]));

disp('SUCCESS')
