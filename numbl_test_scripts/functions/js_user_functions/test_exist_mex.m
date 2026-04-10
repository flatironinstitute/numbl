% Test exist() on .numbl.js user functions (MEX-equivalent → 3)
% and on .m workspace functions (→ 2). This is numbl-specific behavior;
% .numbl.js files are treated like MATLAB MEX files for exist().

% myadd is defined only in myadd.numbl.js → MEX (3)
assert(exist('myadd') == 3, 'exist(myadd) should be 3');
assert(exist('myadd', 'file') == 3, 'exist(myadd, file) should be 3');

% myshadowed has both .m and .numbl.js — .m wins → 2 (matches numbl call resolution)
assert(exist('myshadowed') == 2, 'exist(myshadowed) should be 2');
assert(exist('myshadowed', 'file') == 2, 'exist(myshadowed, file) should be 2');

% Built-in still wins over workspace
assert(exist('sin') == 5, 'exist(sin) should be 5');
assert(exist('sin', 'file') == 0, 'exist(sin, file) should be 0');

% Variable still wins over everything
nothing_var = 7;  %#ok<NASGU>
assert(exist('nothing_var') == 1);
assert(exist('nothing_var', 'var') == 1);

% Search-path lookup: passing the full filename (with extension) should also
% find files anywhere on the search path, not just relative to cwd.
assert(exist('myadd.numbl.js', 'file') == 3, 'should find myadd.numbl.js on search path');
assert(exist('test_js_basic.m', 'file') == 2, 'should find test_js_basic.m on search path');

disp('SUCCESS')
