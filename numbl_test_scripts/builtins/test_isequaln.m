% isequaln: isequal with NaN values treated as equal.

assert(isequaln(NaN, NaN));
assert(~isequal(NaN, NaN));
assert(isequaln([1 NaN 3], [1 NaN 3]));
assert(isequaln([1 NaN], [1 NaN], [1 NaN]));
assert(~isequaln([1 NaN], [1 NaN], [1 2]));
assert(~isequaln(NaN, 1));
assert(~isequaln([NaN NaN], NaN));
assert(isequaln(complex(NaN, 1), complex(NaN, 1)));
assert(~isequaln(complex(1, NaN), complex(NaN, 1)));
assert(isequaln([1+2i NaN], [1+2i NaN]));
assert(isequaln(struct('a', NaN, 'b', 'x'), struct('a', NaN, 'b', 'x')));
assert(~isequal(struct('a', NaN), struct('a', NaN)));
assert(isequaln({NaN, 'a', [1 NaN]}, {NaN, 'a', [1 NaN]}));
assert(isequaln(true, 1));
assert(isequaln('abc', 'abc'));
assert(~isequaln('abc', 'abd'));
try
    isequaln(1);
    error('no error');
catch e
    assert(strcmp(e.message, 'Not enough input arguments.'));
end

disp('SUCCESS');
