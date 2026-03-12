% Test num2str with special values (Inf, -Inf, NaN)

% Inf
assert(strcmp(num2str(Inf), 'Inf'));
assert(strcmp(num2str(-Inf), '-Inf'));

% NaN
assert(strcmp(num2str(NaN), 'NaN'));

% In tensors
v = [1 Inf -Inf NaN 5];
s = num2str(v);
assert(contains(s, 'Inf'));
assert(contains(s, '-Inf'));
assert(contains(s, 'NaN'));

% sprintf also should use Inf not Infinity
assert(strcmp(sprintf('%g', Inf), 'Inf'));
assert(strcmp(sprintf('%g', -Inf), '-Inf'));
assert(strcmp(sprintf('%f', Inf), 'Inf'));
assert(strcmp(sprintf('%f', -Inf), '-Inf'));
assert(strcmp(sprintf('%e', Inf), 'Inf'));
assert(strcmp(sprintf('%e', -Inf), '-Inf'));

disp('SUCCESS');
