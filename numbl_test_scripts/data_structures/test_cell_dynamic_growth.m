% Test cell array dynamic growth via subscript assignment

% Test 1: 2D growth - grow rows
blk = cell(1, 2);
blk{1,1} = 's';
blk{1,2} = 3;
blk{2,1} = 'l';
blk{2,2} = 5;
assert(strcmp(blk{1,1}, 's'));
assert(blk{1,2} == 3);
assert(strcmp(blk{2,1}, 'l'));
assert(blk{2,2} == 5);
[r, c] = size(blk);
assert(r == 2);
assert(c == 2);

% Test 2: 2D growth - grow columns
c2 = cell(2, 1);
c2{1,1} = 10;
c2{1,2} = 20;
c2{2,2} = 30;
assert(c2{1,1} == 10);
assert(c2{1,2} == 20);
assert(c2{2,2} == 30);
assert(isempty(c2{2,1}));
[r2, c2s] = size(c2);
assert(r2 == 2);
assert(c2s == 2);

% Test 3: 2D growth - grow both dimensions
c3 = cell(1, 1);
c3{1,1} = 'a';
c3{3,4} = 'z';
[r3, c3s] = size(c3);
assert(r3 == 3);
assert(c3s == 4);
assert(strcmp(c3{1,1}, 'a'));
assert(strcmp(c3{3,4}, 'z'));
% Intermediate cells should be empty
assert(isempty(c3{2,1}));

% Test 4: Auto-creation from curly-brace assignment
clear;
x{1,1} = 'hello';
x{1,2} = 42;
assert(iscell(x));
assert(strcmp(x{1,1}, 'hello'));
assert(x{1,2} == 42);

% Test 5: 1D growth (should already work)
c5 = cell(1, 2);
c5{1} = 10;
c5{2} = 20;
c5{5} = 50;
assert(c5{1} == 10);
assert(c5{2} == 20);
assert(c5{5} == 50);

disp('SUCCESS');
