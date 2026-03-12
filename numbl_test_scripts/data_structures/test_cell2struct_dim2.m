% Test cell2struct with dim=2

% dim=2: columns map to fields, rows map to struct array elements
c = {'John', 30; 'Jane', 25; 'Bob', 35};
fields = {'name', 'age'};
s = cell2struct(c, fields, 2);
assert(length(s) == 3);
assert(strcmp(s(1).name, 'John'));
assert(s(1).age == 30);
assert(strcmp(s(2).name, 'Jane'));
assert(s(2).age == 25);
assert(strcmp(s(3).name, 'Bob'));
assert(s(3).age == 35);

% dim=2 with single row
c2 = {'Alice', 42};
s2 = cell2struct(c2, fields, 2);
assert(length(s2) == 1);
assert(strcmp(s2.name, 'Alice'));
assert(s2.age == 42);

% dim=2 with more fields
c3 = {'x', 1, true; 'y', 2, false};
f3 = {'label', 'value', 'flag'};
s3 = cell2struct(c3, f3, 2);
assert(length(s3) == 2);
assert(strcmp(s3(1).label, 'x'));
assert(s3(1).value == 1);
assert(s3(1).flag == true);
assert(strcmp(s3(2).label, 'y'));

disp('SUCCESS');
