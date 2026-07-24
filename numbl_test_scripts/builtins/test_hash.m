% hash(hfun, str) — Octave-compatible message digests (RFC test vectors).

assert(strcmp(hash('MD5', ''), 'd41d8cd98f00b204e9800998ecf8427e'));
assert(strcmp(hash('MD5', 'abc'), '900150983cd24fb0d6963f7d28e17f72'));
assert(strcmp(hash('MD5', 'The quick brown fox jumps over the lazy dog'), ...
    '9e107d9d372bb6826bd81d3542a419d6'));

% Case-insensitive algorithm name
assert(strcmp(hash('md5', 'abc'), '900150983cd24fb0d6963f7d28e17f72'));

% A message longer than one 64-byte block
long = repmat('a', 1, 1000);
assert(strcmp(hash('MD5', long), 'cabe45dcc9ae5b66ba86600cca6b8ba8'));

% SHA-256
assert(strcmp(hash('SHA256', ''), ...
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'));
assert(strcmp(hash('SHA256', 'abc'), ...
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'));
assert(strcmp(hash('sha-256', 'abc'), ...
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'));

% Unsupported algorithm errors
ok = false;
try
    hash('CRC32', 'abc');
catch
    ok = true;
end
assert(ok);

disp('SUCCESS');
