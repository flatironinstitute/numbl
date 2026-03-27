% Test tempname and tempdir

% tempname returns a char
t1 = tempname;
assert(ischar(t1));
assert(~isempty(t1));

% tempname with folder argument
t2 = tempname('/tmp');
assert(ischar(t2));
assert(strncmp(t2, '/tmp/', 5));

% two calls return different names
t3 = tempname;
t4 = tempname;
assert(~strcmp(t3, t4));

% tempdir returns a char
td = tempdir;
assert(ischar(td));
assert(~isempty(td));

% tempname with extension
t5 = [tempname, '.dat'];
assert(endsWith(t5, '.dat'));

disp('SUCCESS');
