% Test that warning('off', 'ID') returns a struct with state and identifier
w = warning('off', 'MATLAB:singularMatrix');
assert(isstruct(w));
assert(strcmp(w.state, 'on'));
assert(strcmp(w.identifier, 'MATLAB:singularMatrix'));

% Restoring should be a no-op (no error)
warning(w.state, w.identifier);

% Also test with a different ID
w2 = warning('off', 'MATLAB:nearlySingularMatrix');
assert(isstruct(w2));
assert(strcmp(w2.identifier, 'MATLAB:nearlySingularMatrix'));
warning(w2.state, w2.identifier);

disp('SUCCESS')
