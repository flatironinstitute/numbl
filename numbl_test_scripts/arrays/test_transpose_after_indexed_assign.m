% Test that .' works correctly on a variable that was assigned scalar
% then later grew via indexed assignment (type inference bug)

scl = 1;
if scl == 0
    rows = 0;
else
    rows(1,:) = zeros(1, 5);
    rows(2,:) = [1 2 3 4 5];
    rows(3,:) = [6 7 8 9 10];
end

% rows is [3, 5], transpose should be [5, 3]
rt = rows.';
assert(isequal(size(rt), [5, 3]));
assert(rt(1,1) == 0);
assert(rt(1,2) == 1);
assert(rt(1,3) == 6);

% Same pattern with logical scalar growing via indexed assign
if scl == 0
    flags = true;
else
    flags(1) = true;
    flags(2) = false;
    flags(3) = true;
end
assert(numel(flags) == 3);
assert(flags(1) == 1);
assert(flags(2) == 0);

% Arithmetic on a variable that was scalar but grew to tensor
if scl == 0
    vals = 5;
else
    vals(1) = 10;
    vals(2) = 20;
end
result = vals + 1;
assert(isequal(result, [11 21]));

fprintf('SUCCESS\n');
