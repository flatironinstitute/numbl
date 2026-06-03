% typecast: reinterpret the bytes of a numeric array (numbl treats input as
% double-precision, so this is the serialize-to-bytes direction).

% double -> uint8: 8 little-endian bytes per element
b = typecast(double([1 2]), 'uint8');
assert(numel(b) == 16, 'double->uint8 should give 8 bytes per element');
% 1.0 is 0x3FF0000000000000 -> little-endian bytes end in 240 63
assert(isequal(b(7:8), [240 63]), 'unexpected bytes for 1.0');
% 2.0 is 0x4000000000000000 -> last byte 64
assert(b(16) == 64, 'unexpected high byte for 2.0');

% double -> single: 8 bytes per double reinterpreted as two float32s
s = typecast(double([1 2 3]), 'single');
assert(numel(s) == 6, 'double->single should give 2 float32 per double');

% double -> int32: two int32 per double
i = typecast(double(0), 'int32');
assert(isequal(i, [0 0]), 'double 0 -> two zero int32');

disp('SUCCESS')
