% Test rmfield with a cell array of field names (in addition to a single name).

% Scalar struct, single field name (char).
s.a = 1; s.b = 2; s.c = 3; s.d = 4;
s1 = rmfield(s, 'b');
assert(isequal(sort(fieldnames(s1)), {'a'; 'c'; 'd'}));
assert(s1.a == 1 && s1.c == 3 && s1.d == 4);

% Scalar struct, cell array of names removes them all.
s2 = rmfield(s, {'a', 'c'});
assert(isequal(sort(fieldnames(s2)), {'b'; 'd'}));
assert(s2.b == 2 && s2.d == 4);

% Removing every field leaves an empty-field struct.
s3 = rmfield(s, {'a', 'b', 'c', 'd'});
assert(isempty(fieldnames(s3)));

% Struct array with a cell array of field names.
t(1).x = 10; t(1).y = 'a'; t(1).z = 100;
t(2).x = 20; t(2).y = 'b'; t(2).z = 200;
t2 = rmfield(t, {'y', 'z'});
assert(isequal(fieldnames(t2), {'x'}));
assert(t2(1).x == 10 && t2(2).x == 20);
assert(numel(t2) == 2);

% Removing a nonexistent field errors.
ok = false;
try
    rmfield(s, {'a', 'nope'});
catch
    ok = true;
end
assert(ok, 'rmfield should error on a missing field');

disp('SUCCESS');
