% Warnings issued from compiled code respect the warning state, and the
% state-changing forms inside a function behave as in the interpreter.

warning('off', 'jw:quiet');
assert(emit_warnings(3) == 6);  % shows only the jw:loud warnings
warning('off', 'all');
assert(emit_warnings(2) == 3);  % shows nothing
assert(bare_warning(2) == 4);   % shows nothing
warning('on', 'all');
q = warning('query', 'jw:quiet');
assert(strcmp(q.state, 'on'));

% Turning a warning off inside a function silences it from then on.
assert(toggle_inside(2) == 2);
q = warning('query', 'jw:toggled');
assert(strcmp(q.state, 'off'));

% Char and string identifiers are interchangeable.
warning("off", "jw:str");
q = warning('query', 'jw:str');
assert(strcmp(q.state, 'off'));

warning('off', 'jw:quiet');
warning('off', 'jw:loud');
disp('SUCCESS');
% Everything below is suppressed; any output here fails the test (the
% runner requires SUCCESS to be the last output line).
r = emit_warnings(2);
r = toggle_inside(2);
warning('jw:str', 'string id %d', 1);
warning("jw:str", "string id %d", 2);
warning('off', 'all');
r = bare_warning(2);

function s = emit_warnings(n)
  %!numbl:assert_jit c
  s = 0;
  for k = 1:n
    s = s + k;
    warning('jw:quiet', 'quiet %d', k);
    warning('jw:loud', 'loud %d', k);
  end
end

function s = bare_warning(n)
  %!numbl:assert_jit c
  s = 2 * n;
  warning('bare %d', s);
end

function s = toggle_inside(n)
  s = 0;
  for k = 1:n
    s = s + 1;
    warning('jw:toggled', 'before off %d', k);
    warning('off', 'jw:toggled');
    warning('jw:toggled', 'after off %d', k);
  end
end
