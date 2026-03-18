% Test sparse matrix in conditional (if/while)

%% sparse with zeros should be false
S = sparse([1 0; 0 2]);
if S
    result1 = 'true';
else
    result1 = 'false';
end
assert(strcmp(result1, 'false'));

%% all-nonzero sparse should be true
S2 = sparse([1 2; 3 4]);
if S2
    result2 = 'true';
else
    result2 = 'false';
end
assert(strcmp(result2, 'true'));

%% all-zero sparse should be false
S3 = sparse(2, 2);
if S3
    result3 = 'true';
else
    result3 = 'false';
end
assert(strcmp(result3, 'false'));

disp('SUCCESS')
