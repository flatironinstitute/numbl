% warning('off', id) / warning('on', id) / warning('query', id) track
% per-identifier suppression state, and warning(prevState) restores it.

% Initially on
q = warning('query', 'my:test:id');
assert(strcmp(q.state, 'on'));
assert(strcmp(q.identifier, 'my:test:id'));

% Turn off; returned struct holds the PREVIOUS state
w = warning('off', 'my:test:id');
assert(strcmp(w.state, 'on'));
q2 = warning('query', 'my:test:id');
assert(strcmp(q2.state, 'off'));

% Other ids are unaffected
q3 = warning('query', 'other:id');
assert(strcmp(q3.state, 'on'));

% Restore from the saved struct
warning(w);
q4 = warning('query', 'my:test:id');
assert(strcmp(q4.state, 'on'));

% 'all' suppression covers specific ids; 'on','all' clears it
warning('off', 'all');
q5 = warning('query', 'whatever:id');
assert(strcmp(q5.state, 'off'));
warning('on', 'all');
q6 = warning('query', 'whatever:id');
assert(strcmp(q6.state, 'on'));

% Leave a suppressed id set for the post-SUCCESS check below
warning('off', 'post:success:id');

disp('SUCCESS');
% If suppression is broken, this prints after SUCCESS and the test FAILS
% (the runner requires SUCCESS to be the last output line).
warning('post:success:id', 'this warning must not appear');
