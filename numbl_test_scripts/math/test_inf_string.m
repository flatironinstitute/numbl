% Inf / -Inf must render as 'Inf' / '-Inf' (MATLAB), not the JavaScript
% spelling 'Infinity'. numbl's number formatter sent non-integers through
% toPrecision(5), and Infinity.toPrecision(5) yields "Infinity"; mat2str and
% string() had the same defect.

assert(strcmp(mat2str(Inf), 'Inf'), 'mat2str(Inf)');
assert(strcmp(mat2str(-Inf), '-Inf'), 'mat2str(-Inf)');
assert(strcmp(mat2str([1 Inf 3]), '[1 Inf 3]'), 'mat2str vector with Inf');
assert(strcmp(string(Inf), 'Inf'), 'string(Inf)');
assert(strcmp(string(-Inf), '-Inf'), 'string(-Inf)');

% NaN was already correct -- guard against a regression
assert(strcmp(mat2str(NaN), 'NaN'), 'mat2str(NaN)');

disp('SUCCESS');
