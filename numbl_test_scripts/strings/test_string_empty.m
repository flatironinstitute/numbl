% Test string([]) — an empty string value that concatenation treats as a no-op

out = [string([]), string('gmsh')];
assert(strcmp(class(out), 'string'));
assert(out == "gmsh");

% same pattern via a variable holding an empty double
fmt = [];
s = [string(fmt), "abc"];
assert(s == "abc");

disp('SUCCESS')
