% Test: assigning structs via parenthesis indexing builds a struct array
% This exercises the pattern: newData(k) = someStruct

% Build a struct array by indexed assignment into an uninitialized variable
clear newData;
for k = 1:3
    s.x = k;
    s.y = k * 10;
    newData(k) = s;
end

assert(numel(newData) == 3);
assert(newData(1).x == 1);
assert(newData(2).x == 2);
assert(newData(3).x == 3);
assert(newData(1).y == 10);
assert(newData(2).y == 20);
assert(newData(3).y == 30);

% Also test growing an existing struct array
s2.a = 100;
base2(1) = s2;
s3.a = 200;
base2(2) = s3;
assert(numel(base2) == 2);
assert(base2(1).a == 100);
assert(base2(2).a == 200);

disp('SUCCESS');
