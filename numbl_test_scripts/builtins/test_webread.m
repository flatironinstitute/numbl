% Test webread

% Basic JSON response — should auto-decode to struct
data = webread('https://httpbin.org/get');
assert(isstruct(data));
assert(ischar(data.url));

% With query parameters
data2 = webread('https://httpbin.org/get', 'key1', 'val1', 'key2', 'val2');
assert(isstruct(data2));
assert(strcmp(data2.args.key1, 'val1'));
assert(strcmp(data2.args.key2, 'val2'));

disp('SUCCESS');
