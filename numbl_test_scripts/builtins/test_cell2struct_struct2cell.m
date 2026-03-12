% Test cell2struct and struct2cell

% Basic cell2struct
c = {'John', 30, true};
fields = {'name', 'age', 'active'};
s = cell2struct(c, fields, 2);
assert(strcmp(s.name, 'John'));
assert(s.age == 30);
assert(s.active == true);

% cell2struct with column cell and dim=1
c2 = {'Alice'; 25; false};
s2 = cell2struct(c2, {'name', 'age', 'active'}, 1);
assert(strcmp(s2.name, 'Alice'));
assert(s2.age == 25);

% struct2cell - basic
s3.x = 10;
s3.y = 20;
s3.z = 30;
c3 = struct2cell(s3);
assert(length(c3) == 3);
assert(c3{1} == 10);
assert(c3{2} == 20);
assert(c3{3} == 30);

% Round trip: struct -> cell -> struct
s4.a = 1;
s4.b = 'hello';
c4 = struct2cell(s4);
f4 = fieldnames(s4);
s5 = cell2struct(c4, f4, 1);
assert(s5.a == 1);
assert(strcmp(s5.b, 'hello'));

disp('SUCCESS');
