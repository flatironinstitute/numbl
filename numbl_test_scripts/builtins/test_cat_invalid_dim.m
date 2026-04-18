% cat must reject dim < 1 or non-integer dim
threw = false;
try; cat(0, [1 2], [3 4]); catch; threw = true; end
if ~threw, error('cat(0, ...) should have errored'); end

threw = false;
try; cat(-1, [1 2], [3 4]); catch; threw = true; end
if ~threw, error('cat(-1, ...) should have errored'); end

threw = false;
try; cat(1.5, [1 2], [3 4]); catch; threw = true; end
if ~threw, error('cat(1.5, ...) should have errored'); end

% valid still work
r = cat(1, [1 2], [3 4]);
assert(isequal(r, [1 2; 3 4]));
r = cat(2, [1; 2], [3; 4]);
assert(isequal(r, [1 3; 2 4]));

disp('SUCCESS');
