% A(I) = rhs deletes only when rhs is literally [] (or [ ], ([]), [[]])
% or ''; any other empty value, even a 0x0 double, is an ordinary
% assignment.

mismatch = 'Unable to perform assignment because the left and right sides have a different number of elements.';

% Null literals delete
x = 1:3; x(2) = []; assert(isequal(x, [1 3]));
x = 1:3; x(2) = [ ]; assert(isequal(x, [1 3]));
x = 1:3; x(2) = ([]); assert(isequal(x, [1 3]));
x = 1:3; x(2) = [[]]; assert(isequal(x, [1 3]));
x = 1:3; x(2) = ''; assert(isequal(x, [1 3]));
c = {1, 2, 3}; c(2) = ''; assert(isequal(size(c), [1 2]));
s = 'abc'; s(2) = ''; assert(isequal(s, 'ac'));
q.a = 1:3; q.a(2) = []; assert(isequal(q.a, [1 3]));
st = struct('a', {1, 2, 3}); st(2) = []; assert(isequal(size(st), [1 2]));

% A 0x0 value that is not the literal is assigned
try
    x = 1:3; x(2) = zeros(0, 0); error('no error');
catch e
    assert(strcmp(e.message, mismatch));
end
try
    x = 1:3; e0 = []; x(2) = e0; error('no error');
catch e
    assert(strcmp(e.message, mismatch));
end
try
    x = 1:3; x(2) = [[], []]; error('no error');
catch e
    assert(strcmp(e.message, mismatch));
end
try
    q.a = 1:3; q.a(2) = zeros(0, 0); error('no error');
catch e
    assert(strcmp(e.message, mismatch));
end
threw = false;
try
    c = {1, 2, 3}; c(2) = zeros(0, 0);
catch
    threw = true;
end
assert(threw);
try
    st = struct('a', {1, 2, 3}); st(2) = zeros(0, 0); error('no error');
catch e
    assert(strcmp(e.message, 'Assignment between unlike types is not allowed.'));
end

% Assigning an empty value to an empty selection is a no-op that can grow
F = zeros(2, 3); F(5, 1:0) = zeros(0, 0); assert(isequal(size(F), [5 3]));
G = zeros(2, 3); G(1:0, 7) = zeros(0, 0); assert(isequal(size(G), [2 7]));
x = 1:3; x(1:0) = zeros(0, 0); assert(isequal(x, 1:3));
x = 1:3; x([]) = zeros(0, 0); assert(isequal(x, 1:3));

% A null assignment allows one non-colon subscript, even when the
% subscripts select nothing
bad = {'G(1:0, 7) = [];', 'G(1:0, 2) = [];', 'G([], []) = [];', ...
       'G(1:0, 1:0) = [];', 'G(false(2, 1), 2) = [];'};
for k = 1:numel(bad)
    G = zeros(2, 3);
    try
        eval(bad{k});
        error('no error');
    catch e
        assert(strcmp(e.message, 'A null assignment can have only one non-colon index.'));
    end
end
G = zeros(2, 3); G(:, []) = []; assert(isequal(size(G), [2 3]));
G = zeros(2, 3); G(1:0, :) = []; assert(isequal(size(G), [2 3]));

% subsasgn deletes for any 0x0 double value
x = 1:3; e0 = zeros(0, 0);
x = subsasgn(x, substruct('()', {2}), e0); assert(isequal(x, [1 3]));

disp('SUCCESS');
