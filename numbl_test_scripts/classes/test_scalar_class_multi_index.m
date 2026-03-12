% Test that scalar class instance indexing with multiple indices works
% e.g., f(:,1), f(1,:), f(1,1) on a scalar object via builtin('subsref')

obj = ScalarMultiIdx_(42);

% Test basic property access
assert(obj.val == 42);

% Test f(:,1) - colon + scalar index via builtin subsref
r1 = obj(:,1);
assert(r1.val == 42);

% Test f(1,:)
r2 = obj(1,:);
assert(r2.val == 42);

% Test f(1,1)
r3 = obj(1,1);
assert(r3.val == 42);

fprintf('SUCCESS\n');
