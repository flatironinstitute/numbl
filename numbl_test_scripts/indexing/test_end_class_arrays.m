% `end` on class instances and object arrays: an overloaded end method is
% called (for scalars and arrays alike); otherwise the builtin dimensions
% are used, ignoring any size/numel overloads.

% No end overload: builtin dims, not the overloaded size/numel
o = EndSizeOverload_(4);
r = o(end);
assert(r.v == 4);
a = [EndSizeOverload_(1) EndSizeOverload_(2) EndSizeOverload_(3)];
r = a(end);
assert(r.v == 3);
r = a(1, end);
assert(r.v == 3);
p = [EndSizeOverload_(1) EndSizeOverload_(2); EndSizeOverload_(3) EndSizeOverload_(4)];
r = p(end, 1);
assert(r.v == 3);
r = p(end);
assert(r.v == 4);
a(end+1) = EndSizeOverload_(5);
r = a(4);
assert(r.v == 5);

% end overload (returns the number of subscripts)
e = [EndArgsOverload_(1) EndArgsOverload_(2) EndArgsOverload_(3)];
r = e(end);
assert(r.v == 1);
r = e(1, end);
assert(r.v == 2);
b = [EndArgsOverload_(1); EndArgsOverload_(2); EndArgsOverload_(3)];
r = b(end, 1);
assert(r.v == 2);
q = EndArgsOverload_(9);
r = q(end);
assert(r.v == 9);

disp('SUCCESS');
