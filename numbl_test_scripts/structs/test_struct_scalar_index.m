% A scalar (1x1) struct accepts any all-ones subscripting and returns itself.
s.a = 1;
s.b = 2;
assert(s(1).a == 1, 's(1)');
assert(s(1, 1).b == 2, 's(1,1)');
assert(s(1, 1, 1).a == 1, 's(1,1,1)');

% Out-of-range subscript errors.
err = false;
try
    t = s(2);
catch
    err = true;
end
assert(err, 's(2) should error');
disp('SUCCESS');
