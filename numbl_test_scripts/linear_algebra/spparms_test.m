% spparms: sparse-solver parameter stub. numbl ignores the knobs, but the
% values round-trip through get/set/restore (the pattern ultraSEM relies on).
% Defaults and key order verified against MATLAB R2025b.

p = spparms;
assert(isequal(size(p), [19 1]), 'spparms returns a 19x1 vector');

% bandden is the 11th parameter; default 0.5.
assert(spparms('bandden') == 0.5, 'default bandden should be 0.5');

% Round-trip: set then restore from the saved vector.
spparms('bandden', 0);
assert(spparms('bandden') == 0, 'bandden should be 0 after set');
spparms(p);
assert(spparms('bandden') == 0.5, 'bandden restored to 0.5');

% [keys, vals] form.
[keys, vals] = spparms;
assert(size(keys, 1) == 19, 'keys should have 19 rows');
assert(isequal(size(vals), [19 1]), 'vals should be 19x1');
assert(strcmp(strtrim(keys(11, :)), 'bandden'), '11th key should be bandden');

% 'default' preset resets all parameters.
spparms('autoamd', 0);
spparms('default');
assert(spparms('autoamd') == 1, 'default preset resets autoamd to 1');

disp('SUCCESS')
