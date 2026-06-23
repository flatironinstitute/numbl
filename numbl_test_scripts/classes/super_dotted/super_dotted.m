% Regression test: superclass constructor / method calls where the
% superclass is package-qualified (dotted), e.g. obj@shp.Base(...) and
% describe@shp.Base(...). Previously the parser only consumed a single
% identifier after '@', so a dotted superclass name was a syntax error.
% (This is the pattern ultraSEM uses: obj@ultraSEM.Quad(...).)

d = shp.Derived(5, 7);
assert(d.v == 5, 'superclass constructor should set v=5');
assert(d.w == 7, 'derived constructor should set w=7');

% describe@shp.Base(obj) must call the inherited Base.describe.
s = d.describe();
assert(strcmp(s, 'Base(5)+Derived(7)'), ['describe gave: ' s]);

% Default construction path (no args -> empty superclass args via {:}).
d0 = shp.Derived();
assert(d0.v == 0 && d0.w == 0, 'default construction');

disp('SUCCESS')
