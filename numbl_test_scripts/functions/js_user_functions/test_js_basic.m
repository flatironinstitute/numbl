% Test basic .js user function calls
result = myadd(3, 4);
assert(result == 7);

result2 = mymul(5, 6);
assert(result2 == 30);

% Compose js user functions
result3 = myadd(mymul(2, 3), mymul(4, 5));
assert(result3 == 26);

disp('SUCCESS')
