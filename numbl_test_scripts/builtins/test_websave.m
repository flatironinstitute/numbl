% Test websave

% Basic download
f = websave('/tmp/numbl_test_websave.txt', 'https://httpbin.org/get');
assert(ischar(f));
assert(strcmp(f, '/tmp/numbl_test_websave.txt'));
content = fileread(f);
assert(~isempty(content));
assert(contains(content, '"url"'));

% With query parameters
f2 = websave('/tmp/numbl_test_websave2.txt', 'https://httpbin.org/get', 'key1', 'val1');
content2 = fileread(f2);
assert(contains(content2, 'key1'));
assert(contains(content2, 'val1'));

disp('SUCCESS');
