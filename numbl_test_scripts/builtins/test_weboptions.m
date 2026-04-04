% Test weboptions

% Default weboptions
opts = weboptions;
assert(isstruct(opts));
assert(opts.Timeout == 5);
assert(strcmp(opts.CharacterEncoding, 'auto'));
assert(strcmp(opts.UserAgent, 'numbl'));
assert(strcmp(opts.RequestMethod, 'auto'));
assert(strcmp(opts.ContentType, 'auto'));
assert(strcmp(opts.MediaType, 'auto'));
assert(strcmp(opts.ArrayFormat, 'csv'));
assert(strcmp(opts.CertificateFilename, 'default'));

% Custom timeout
opts2 = weboptions('Timeout', 60);
assert(opts2.Timeout == 60);
assert(strcmp(opts2.RequestMethod, 'auto'));

% Multiple options
opts3 = weboptions('Timeout', 30, 'RequestMethod', 'post');
assert(opts3.Timeout == 30);
assert(strcmp(opts3.RequestMethod, 'post'));

% websave with weboptions
opts4 = weboptions('Timeout', 60);
f = websave('/tmp/numbl_test_weboptions.txt', 'https://httpbin.org/get', opts4);
content = fileread(f);
assert(contains(content, '"url"'));

% webread with weboptions
opts5 = weboptions('Timeout', 60);
data = webread('https://httpbin.org/get', opts5);
assert(isstruct(data));

disp('SUCCESS');
