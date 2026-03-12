%% strcmp - basic char comparison
assert(strcmp('hello', 'hello') == true);
assert(strcmp('hello', 'world') == false);
assert(strcmp('Yes', 'yes') == false);  % case sensitive

%% strcmpi - case insensitive
assert(strcmpi('Yes', 'yes') == true);
assert(strcmpi('Yes', 'No') == false);

%% strcmp with unsupported data types returns 0
f = @(x) x + 1;
assert(strcmp(f, 'hello') == false);
assert(strcmp('hello', f) == false);
assert(strcmp(42, 'hello') == false);
assert(strcmp('hello', 42) == false);

%% strcmpi with unsupported data types returns 0
assert(strcmpi(f, 'hello') == false);
assert(strcmpi('hello', f) == false);
assert(strcmpi(42, 'Hello') == false);
assert(strcmpi('Hello', 42) == false);

%% strcmp - scalar vs cell array
s1 = 'once';
s2 = {'Once', 'upon'; 'a', 'time'};
tf = strcmp(s1, s2);
assert(isequal(size(tf), [2 2]));
assert(tf(1,1) == false);  % case sensitive: 'once' ~= 'Once'
assert(tf(1,2) == false);
assert(tf(2,1) == false);
assert(tf(2,2) == false);

%% strcmpi - scalar vs cell array (case insensitive)
tf2 = strcmpi(s1, s2);
assert(isequal(size(tf2), [2 2]));
assert(tf2(1,1) == true);  % 'once' == 'Once' ignoring case
assert(tf2(1,2) == false);
assert(tf2(2,1) == false);
assert(tf2(2,2) == false);

%% strcmp - cell array vs cell array (same size)
s3 = {'Tinker', 'Tailor'; '  Soldier', 'Spy'};
s4 = {'Tinker', 'Baker'; 'Soldier', 'Spy'};
tf3 = strcmp(s3, s4);
assert(isequal(size(tf3), [2 2]));
assert(tf3(1,1) == true);   % 'Tinker' == 'Tinker'
assert(tf3(1,2) == false);  % 'Tailor' ~= 'Baker'
assert(tf3(2,1) == false);  % '  Soldier' ~= 'Soldier' (whitespace)
assert(tf3(2,2) == true);   % 'Spy' == 'Spy'

%% strcmpi - cell array vs cell array (case insensitive)
s5 = {'Tinker', 'Tailor'; '  Soldier', 'Spy'};
s6 = {'Tinker', 'Baker'; 'Soldier', 'SPY'};
tf4 = strcmpi(s5, s6);
assert(isequal(size(tf4), [2 2]));
assert(tf4(1,1) == true);
assert(tf4(1,2) == false);
assert(tf4(2,1) == false);  % whitespace still matters
assert(tf4(2,2) == true);   % 'Spy' == 'SPY' ignoring case

disp('SUCCESS')
