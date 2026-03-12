% Test dec2hex, hex2dec, dec2bin, bin2dec

%% dec2hex
assert(strcmp(dec2hex(255), 'FF'));
assert(strcmp(dec2hex(0), '0'));
assert(strcmp(dec2hex(16), '10'));
assert(strcmp(dec2hex(10), 'A'));

%% dec2hex with minimum digits
assert(strcmp(dec2hex(10, 4), '000A'));

%% hex2dec
assert(hex2dec('FF') == 255);
assert(hex2dec('ff') == 255);
assert(hex2dec('0') == 0);
assert(hex2dec('10') == 16);

%% dec2bin
assert(strcmp(dec2bin(5), '101'));
assert(strcmp(dec2bin(0), '0'));
assert(strcmp(dec2bin(8), '1000'));
assert(strcmp(dec2bin(1), '1'));

%% dec2bin with minimum digits
assert(strcmp(dec2bin(5, 8), '00000101'));

%% bin2dec
assert(bin2dec('101') == 5);
assert(bin2dec('0') == 0);
assert(bin2dec('1000') == 8);
assert(bin2dec('11111111') == 255);

disp('SUCCESS');
