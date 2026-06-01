% Test: external method file whose internal function name differs from the
% file name. Dispatch is by file name (bump), so bump(obj) must resolve to
% the file's primary function even though it is declared `differentName`.

obj = NameMismatch_(5);

% Function-call form
r1 = bump(obj);
assert(r1.value == 105, 'Expected 105 from bump(obj)');

% Method-call (dot) form
r2 = obj.bump();
assert(r2.value == 105, 'Expected 105 from obj.bump()');

disp('SUCCESS')
