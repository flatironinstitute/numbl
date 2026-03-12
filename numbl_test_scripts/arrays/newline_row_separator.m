% Test: Newlines inside [] should act as row separators

% Basic 3x3 matrix with newline row separators
M1 = [1 2 3
      4 5 6
      7 8 9];
assert(size(M1, 1) == 3);
assert(size(M1, 2) == 3);
assert(M1(2, 2) == 5);
assert(M1(3, 1) == 7);

% Column vector with newlines
c = [1
     2
     3];
assert(size(c, 1) == 3);
assert(size(c, 2) == 1);

% With commas
M2 = [1, 2, 3
      4, 5, 6];
assert(size(M2, 1) == 2);
assert(size(M2, 2) == 3);
assert(M2(2, 1) == 4);

% Mixed semicolons and newlines
M3 = [1 2; 3 4
      5 6];
assert(size(M3, 1) == 3);
assert(size(M3, 2) == 2);
assert(M3(3, 1) == 5);

disp('SUCCESS');
