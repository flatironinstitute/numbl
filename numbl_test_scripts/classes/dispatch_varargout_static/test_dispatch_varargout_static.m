% Test: method dispatch on objects returned from static varargout methods
%
% When a static method uses varargout, the return type should be Unknown
% (not Cell), so that method calls on the returned object are deferred to
% runtime dispatch rather than generating a compile-time "Unknown function" error.

% Single output from varargout static method
v = VarargOutVec.create([1 2 3]);
assert(isa(v, 'VarargOutVec'));

% Call an instance method on the returned object — this is the key test.
% Before the fix, `scale(v, 2)` would fail with "Unknown function: scale"
% because the compiler inferred v's type as Cell (the type of varargout)
% instead of Unknown.
w = scale(v, 2);
assert(isa(w, 'VarargOutVec'));
assert(isequal(w.data, [2 4 6]));

% Multiple outputs from varargout static method
[a, b] = VarargOutVec.create([10 20], [30 40]);
assert(isa(a, 'VarargOutVec'));
assert(isa(b, 'VarargOutVec'));
wa = scale(a, 3);
wb = scale(b, 5);
assert(isequal(wa.data, [30 60]));
assert(isequal(wb.data, [150 200]));

disp('SUCCESS');
