% Regression: mat2str on complex values used to drop the imaginary part
% entirely — mat2str([1+2i 3-4i]) returned "[1 3]" (a valid-looking but
% numerically wrong string), and a complex scalar errored ("does not support
% these argument types"). resolve() rejected complex_or_number and fmt()
% read only the real lane. Now complex scalars and tensors format as MATLAB
% does: re+imi / re-imi, with an explicit imaginary part.

assert(strcmp(mat2str([1+2i 3-4i]), '[1+2i 3-4i]'), 'complex row vector');
assert(strcmp(mat2str(3+4i), '3+4i'), 'complex scalar +');
assert(strcmp(mat2str(1-2i), '1-2i'), 'complex scalar -');
assert(strcmp(mat2str(2i), '0+2i'), 'pure imaginary');
assert(strcmp(mat2str(-3i), '0-3i'), 'pure negative imaginary');
assert(strcmp(mat2str([1+1i; 2-2i]), '[1+1i;2-2i]'), 'complex column vector');

% Real inputs are unchanged.
assert(strcmp(mat2str([1.5 2; 3 4]), '[1.5 2;3 4]'), 'real matrix');
assert(strcmp(mat2str(5), '5'), 'real scalar');
assert(strcmp(mat2str([1 2 3]), '[1 2 3]'), 'real row vector');

disp('SUCCESS')
