% MATLAB (undocumented) allows addOptional arguments to be passed as
% name-value pairs; real-world code (e.g. Pulseq) relies on this.

p = inputParser;
addRequired(p, 'flip', @isnumeric);
addOptional(p, 'system', [], @isstruct);
addParamValue(p, 'duration', 0, @isnumeric);
addParamValue(p, 'use', 'u');

sys.gamma = 42.58e6;

% Optional passed by name, mixed with parameters
parse(p, 0.5, 'duration', 3e-3, 'system', sys, 'use', 'excitation');
assert(p.Results.flip == 0.5);
assert(p.Results.duration == 3e-3);
assert(isstruct(p.Results.system));
assert(p.Results.system.gamma == 42.58e6);
assert(strcmp(p.Results.use, 'excitation'));

% Optional passed by name is not in UsingDefaults
assert(~any(strcmp(p.UsingDefaults, 'system')));

% Optional still works positionally
p2 = inputParser;
addRequired(p2, 'flip');
addOptional(p2, 'system', []);
addParamValue(p2, 'duration', 0);
parse(p2, 0.5, sys, 'duration', 1e-3);
assert(isstruct(p2.Results.system));
assert(p2.Results.duration == 1e-3);

% Optional omitted entirely -> default, listed in UsingDefaults
p3 = inputParser;
addRequired(p3, 'flip');
addOptional(p3, 'system', [], @isstruct);
addParamValue(p3, 'duration', 0);
parse(p3, 0.5, 'duration', 2e-3);
assert(isempty(p3.Results.system));
assert(any(strcmp(p3.UsingDefaults, 'system')));

% An optional name also terminates positional consumption for a later
% optional (the name is recognized, not consumed as a positional value)
p4 = inputParser;
addRequired(p4, 'n');
addOptional(p4, 'a', -1);
addOptional(p4, 'b', -2);
parse(p4, 7, 'b', 5);
assert(p4.Results.a == -1);
assert(p4.Results.b == 5);

disp('SUCCESS');
