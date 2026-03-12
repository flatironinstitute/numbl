% Test using MATLAB keywords as struct field names
% In MATLAB, keywords can be used as field names after a dot

% Reading keyword fields
s = struct('import', 1, 'for', 2, 'if', 3, 'while', 4);
assert(s.import == 1);
assert(s.for == 2);
assert(s.if == 3);
assert(s.while == 4);

% Writing keyword fields via dot assignment
s2 = struct();
s2.switch = 10;
s2.case = 20;
s2.break = 30;
s2.continue = 40;
s2.return = 50;
s2.try = 60;
s2.catch = 70;
s2.global = 80;
s2.persistent = 90;
s2.function = 100;
s2.otherwise = 110;
s2.true = 120;
s2.false = 130;
s2.end = 140;
s2.classdef = 150;
s2.properties = 160;
s2.methods = 170;
s2.events = 180;
s2.arguments = 190;

assert(s2.switch == 10);
assert(s2.case == 20);
assert(s2.break == 30);
assert(s2.continue == 40);
assert(s2.return == 50);
assert(s2.try == 60);
assert(s2.catch == 70);
assert(s2.global == 80);
assert(s2.persistent == 90);
assert(s2.function == 100);
assert(s2.otherwise == 110);
assert(s2.true == 120);
assert(s2.false == 130);
assert(s2.end == 140);
assert(s2.classdef == 150);
assert(s2.properties == 160);
assert(s2.methods == 170);
assert(s2.events == 180);
assert(s2.arguments == 190);

disp('SUCCESS');
