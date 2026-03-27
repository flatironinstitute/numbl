% Test fread / fwrite binary I/O

% --- Test 1: Write and read uint8 (default precision) ---
fid = fopen('_test_fread_tmp.bin', 'w');
fwrite(fid, [1 2 3 4 5 6 7 8 9]);
fclose(fid);

fid = fopen('_test_fread_tmp.bin', 'r');
A = fread(fid);
fclose(fid);
assert(isequal(A, (1:9)'), 'Test 1 failed: default uint8 read');

% --- Test 2: Read with sizeA ---
fid = fopen('_test_fread_tmp.bin', 'r');
A = fread(fid, [3, 3]);
fclose(fid);
assert(isequal(size(A), [3 3]), 'Test 2 failed: size mismatch');
assert(isequal(A, [1 4 7; 2 5 8; 3 6 9]), 'Test 2 failed: values mismatch');

% --- Test 3: Read with count ---
fid = fopen('_test_fread_tmp.bin', 'r');
A = fread(fid, 5);
fclose(fid);
assert(isequal(A, (1:5)'), 'Test 3 failed: count read');

% --- Test 4: Write/read double precision ---
fid = fopen('_test_fread_double.bin', 'w');
fwrite(fid, [8 3 4; 1 5 9; 6 7 2], 'double');
fclose(fid);

fid = fopen('_test_fread_double.bin', 'r');
A = fread(fid, [3 3], 'double');
fclose(fid);
assert(isequal(A, [8 3 4; 1 5 9; 6 7 2]), 'Test 4 failed: double read');

% --- Test 5: Write/read uint16 ---
fid = fopen('_test_fread_u16.bin', 'w');
fwrite(fid, [1:9], 'uint16');
fclose(fid);

fid = fopen('_test_fread_u16.bin', 'r');
A = fread(fid, [3, 2], 'uint16');
fclose(fid);
assert(isequal(A, [1 4; 2 5; 3 6]), 'Test 5 failed: uint16 read');

% --- Test 6: Read with [A, count] output ---
fid = fopen('_test_fread_tmp.bin', 'r');
[A, count] = fread(fid);
fclose(fid);
assert(count == 9, 'Test 6 failed: count mismatch');
assert(isequal(A, (1:9)'), 'Test 6 failed: values mismatch');

% --- Test 7: Write/read int32 ---
fid = fopen('_test_fread_i32.bin', 'w');
fwrite(fid, [-100 0 100 200], 'int32');
fclose(fid);

fid = fopen('_test_fread_i32.bin', 'r');
A = fread(fid, 4, 'int32');
fclose(fid);
assert(isequal(A, [-100; 0; 100; 200]), 'Test 7 failed: int32 read');

% --- Test 8: Write/read single (float32) ---
fid = fopen('_test_fread_f32.bin', 'w');
fwrite(fid, [1.5 2.5 3.5], 'single');
fclose(fid);

fid = fopen('_test_fread_f32.bin', 'r');
A = fread(fid, 3, 'single');
fclose(fid);
assert(abs(A(1) - 1.5) < 1e-6, 'Test 8 failed: single read');
assert(abs(A(2) - 2.5) < 1e-6, 'Test 8 failed: single read');
assert(abs(A(3) - 3.5) < 1e-6, 'Test 8 failed: single read');

% --- Test 9: frewind ---
fid = fopen('_test_fread_tmp.bin', 'r');
A1 = fread(fid, 3);
frewind(fid);
A2 = fread(fid, 3);
fclose(fid);
assert(isequal(A1, A2), 'Test 9 failed: frewind');

% --- Test 10: fseek + ftell ---
fid = fopen('_test_fread_tmp.bin', 'r');
fseek(fid, 3, 'bof');
assert(ftell(fid) == 3, 'Test 10 failed: ftell after seek');
A = fread(fid, 3);
assert(isequal(A, [4; 5; 6]), 'Test 10 failed: read after seek');
fclose(fid);

% --- Test 11: *uint8 precision (output stays uint8 class, but we return double) ---
fid = fopen('_test_fread_tmp.bin', 'r');
A = fread(fid, 3, '*uint8');
fclose(fid);
assert(isequal(A, [1; 2; 3]), 'Test 11 failed: *uint8 read');

% --- Test 12: *char precision returns char array ---
fid = fopen('_test_fread_char.bin', 'w');
fwrite(fid, 'Hello World');
fclose(fid);

fid = fopen('_test_fread_char.bin', 'r');
txt = fread(fid, '*char')';
fclose(fid);
assert(ischar(txt), 'Test 12 failed: should be char');
assert(strcmp(txt, 'Hello World'), 'Test 12 failed: char content mismatch');

% --- Cleanup ---
delete('_test_fread_tmp.bin');
delete('_test_fread_double.bin');
delete('_test_fread_u16.bin');
delete('_test_fread_i32.bin');
delete('_test_fread_f32.bin');
delete('_test_fread_char.bin');

disp('SUCCESS');
