% Test that cell indexing with a variable vector index produces a CSL
% that can be passed to functions like cat()

% Basic case: cat with cell indexing using variable vector index
c = {[1; 2], [3; 4], [5; 6]};
idx = [1 2];
result = cat(1, c{idx});
assert(isequal(result, [1; 2; 3; 4]));

% All indices
idx = [1 2 3];
result = cat(1, c{idx});
assert(isequal(result, [1; 2; 3; 4; 5; 6]));

% Single index (should still work)
idx = 2;
result = cat(1, c{idx});
assert(isequal(result, [3; 4]));

% Test with other functions that accept variable args
c2 = {[1 0; 0 1], [2 0; 0 2]};
idx = [1 2];
result = blkdiag(c2{idx});
assert(isequal(result, [1 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 2]));

fprintf('SUCCESS\n');
