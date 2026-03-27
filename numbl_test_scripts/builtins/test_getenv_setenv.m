% Test getenv and setenv builtins

% setenv then getenv
setenv('NUMBL_TEST_VAR', 'hello_world');
val = getenv('NUMBL_TEST_VAR');
assert(strcmp(val, 'hello_world'), 'getenv should return set value');

% setenv overwrites
setenv('NUMBL_TEST_VAR', 'updated');
val = getenv('NUMBL_TEST_VAR');
assert(strcmp(val, 'updated'), 'getenv should return updated value');

% getenv for non-existent variable returns empty char
val = getenv('NUMBL_NONEXISTENT_VAR_12345');
assert(ischar(val), 'getenv should return char');
assert(isempty(val), 'getenv should return empty for non-existent var');

% setenv(varname) sets to empty
setenv('NUMBL_TEST_VAR');
val = getenv('NUMBL_TEST_VAR');
assert(strcmp(val, ''), 'setenv with one arg should set to empty');

% getenv() returns dictionary
setenv('NUMBL_TEST_DICT', 'test_value');
d = getenv();
assert(strcmp(class(d), 'dictionary'), 'getenv() should return a dictionary');
assert(isKey(d, "NUMBL_TEST_DICT"), 'dictionary should contain set var');
assert(d("NUMBL_TEST_DICT") == "test_value", 'dictionary value should match');

% setenv(d) — set from dictionary
d2 = dictionary;
d2("NUMBL_A") = "val_a";
d2("NUMBL_B") = "val_b";
setenv(d2);
assert(strcmp(getenv('NUMBL_A'), 'val_a'), 'setenv(d) should set NUMBL_A');
assert(strcmp(getenv('NUMBL_B'), 'val_b'), 'setenv(d) should set NUMBL_B');

% Clean up
setenv('NUMBL_TEST_VAR');
setenv('NUMBL_TEST_DICT');
setenv('NUMBL_A');
setenv('NUMBL_B');

disp('SUCCESS');
