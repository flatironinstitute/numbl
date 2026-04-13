% Test validateattributes

% --- Class validation ---
validateattributes(5, {'numeric'}, {});
validateattributes(5, {'double'}, {});
validateattributes('hello', {'char'}, {});
validateattributes(true, {'logical'}, {});
validateattributes({1, 2}, {'cell'}, {});

% Multiple classes - match any
validateattributes(5, {'char', 'numeric'}, {});

% Class mismatch
try
    validateattributes('hello', {'numeric'}, {});
    error('Should have failed');
catch e
    assert(contains(e.message, 'type'));
end

% Empty classes - skip class check
validateattributes('hello', {}, {'nonempty'});

% --- Scalar ---
validateattributes(5, {'numeric'}, {'scalar'});
try
    validateattributes([1 2 3], {'numeric'}, {'scalar'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'scalar'));
end

% --- Vector ---
validateattributes([1 2 3], {'numeric'}, {'vector'});
validateattributes([1; 2; 3], {'numeric'}, {'vector'});
validateattributes(5, {'numeric'}, {'vector'});
try
    validateattributes([1 2; 3 4], {'numeric'}, {'vector'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'vector'));
end

% --- Row / Column ---
validateattributes([1 2 3], {'numeric'}, {'row'});
validateattributes([1; 2; 3], {'numeric'}, {'column'});
try
    validateattributes([1; 2], {'numeric'}, {'row'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'row'));
end

% --- Nonempty ---
validateattributes(5, {'numeric'}, {'nonempty'});
try
    validateattributes([], {'numeric'}, {'nonempty'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'nonempty'));
end

% --- Size ---
validateattributes(ones(3,4), {'numeric'}, {'size', [3, 4]});
try
    validateattributes(ones(3,5), {'numeric'}, {'size', [3, 4]});
    error('Should have failed');
catch e
    assert(contains(e.message, 'size'));
end

% Size with NaN (skip dimension)
validateattributes(ones(3,5), {'numeric'}, {'size', [3, NaN]});
validateattributes(ones(3,5), {'numeric'}, {'size', [NaN, 5]});

% --- 2d ---
validateattributes(ones(3,4), {'numeric'}, {'2d'});
validateattributes(5, {'numeric'}, {'2d'});

% --- Square ---
validateattributes(ones(3,3), {'numeric'}, {'square'});
try
    validateattributes(ones(3,4), {'numeric'}, {'square'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'square'));
end

% --- Diagonal ---
validateattributes(eye(3), {'numeric'}, {'diag'});
validateattributes(5, {'numeric'}, {'diag'});
try
    validateattributes(ones(3,3), {'numeric'}, {'diag'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'diagonal'));
end

% --- numel, nrows, ncols ---
validateattributes(ones(3,4), {'numeric'}, {'numel', 12});
validateattributes(ones(3,4), {'numeric'}, {'nrows', 3});
validateattributes(ones(3,4), {'numeric'}, {'ncols', 4});
try
    validateattributes(ones(3,4), {'numeric'}, {'numel', 10});
    error('Should have failed');
catch e
    assert(contains(e.message, 'elements'));
end

% --- Range checks ---
validateattributes([5 6 7], {'numeric'}, {'>', 4});
validateattributes([1 2 3], {'numeric'}, {'>=', 1});
validateattributes([1 2 3], {'numeric'}, {'<', 4});
validateattributes([1 2 3], {'numeric'}, {'<=', 3});
validateattributes([1 2 3], {'numeric'}, {'>=', 1, '<=', 3});

try
    validateattributes([5 6 7], {'numeric'}, {'>', 6});
    error('Should have failed');
catch e
    assert(contains(e.message, '>'));
end

% --- Finite / NonNaN ---
validateattributes([1 2 3], {'numeric'}, {'finite'});
try
    validateattributes([1 Inf 3], {'numeric'}, {'finite'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'finite'));
end

validateattributes([1 2 3], {'numeric'}, {'nonnan'});
try
    validateattributes([1 NaN 3], {'numeric'}, {'nonnan'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'NaN'));
end

% --- Integer ---
validateattributes([1 2 3], {'numeric'}, {'integer'});
try
    validateattributes([1.5], {'numeric'}, {'integer'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'integer'));
end

% --- Positive / Nonnegative / Nonzero ---
validateattributes([1 2 3], {'numeric'}, {'positive'});
validateattributes([0 1 2], {'numeric'}, {'nonnegative'});
validateattributes([1 2 3], {'numeric'}, {'nonzero'});

try
    validateattributes([-1], {'numeric'}, {'positive'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'positive'));
end

try
    validateattributes([-1], {'numeric'}, {'nonnegative'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'nonnegative'));
end

try
    validateattributes([0], {'numeric'}, {'nonzero'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'nonzero'));
end

% --- Binary ---
validateattributes([0 1 0 1], {'numeric'}, {'binary'});
try
    validateattributes([0 1 2], {'numeric'}, {'binary'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'binary'));
end

% --- Even / Odd ---
validateattributes([2 4 6], {'numeric'}, {'even'});
validateattributes([1 3 5], {'numeric'}, {'odd'});
try
    validateattributes([1 2 3], {'numeric'}, {'even'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'even'));
end

% --- Increasing / Nondecreasing ---
validateattributes([1; 5; 9], {'numeric'}, {'increasing'});
validateattributes([1; 5; 5], {'numeric'}, {'nondecreasing'});
try
    validateattributes([1; 5; 5], {'numeric'}, {'increasing'});
    error('Should have failed');
catch e
    assert(contains(e.message, 'increasing'));
end

% Matrix column-wise monotonicity
A = [1 5 8 2; 9 6 9 4];
validateattributes(A, {'double'}, {'increasing'});
validateattributes(A, {'double'}, {'nondecreasing'});

% --- Decreasing / Nonincreasing ---
validateattributes([9; 5; 1], {'numeric'}, {'decreasing'});
validateattributes([9; 5; 5], {'numeric'}, {'nonincreasing'});

% --- Multiple attributes ---
validateattributes([1 2 3], {'numeric'}, {'row', 'nonempty', 'positive', 'integer'});

% --- Optional args: argIndex ---
try
    validateattributes('hello', {'numeric'}, {}, 2);
    error('Should have failed');
catch e
    assert(contains(e.message, 'number 2'));
end

% --- Optional args: funcName, varName, argIndex ---
try
    validateattributes('hello', {'numeric'}, {}, 'myFunc', 'myVar', 1);
    error('Should have failed');
catch e
    assert(contains(e.message, 'myVar'));
    assert(contains(e.message, 'number 1'));
end

% --- Real ---
validateattributes([1 2 3], {'numeric'}, {'real'});

% --- ndims ---
validateattributes(ones(3,4), {'numeric'}, {'ndims', 2});

disp('SUCCESS');
