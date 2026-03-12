% Minimal test for method return type inference.
% When a method returns a class instance, the compiler should track
% that type so subsequent method calls use direct dispatch rather
% than falling back to $rt.methodCall.

p = Pair_(3, 7);
assert(strcmp(__inferred_type_str(p), 'ClassInstance<Pair_>'));

% Direct method call — type is known, should be direct dispatch
assert(p.sum() == 10);

% Method returning same class — return type should be ClassInstance<Pair_>
p2 = p.swap();
assert(strcmp(__inferred_type_str(p2), 'ClassInstance<Pair_>'));
assert(p2.A == 7);
assert(p2.B == 3);

% Call method on result — should also be direct dispatch (not $rt.methodCall)
assert(p2.sum() == 10);

% Chain: swap returns Pair_, so swap().sum() should both be direct
assert(strcmp(__inferred_type_str(p.swap()), 'ClassInstance<Pair_>'));
assert(p.swap().sum() == 10);

% Method returning self type through a helper that calls constructor
p3 = p.doubled();
assert(strcmp(__inferred_type_str(p3), 'ClassInstance<Pair_>'));
assert(p3.A == 6);
assert(p3.B == 14);
assert(p3.sum() == 20);

disp('SUCCESS')
