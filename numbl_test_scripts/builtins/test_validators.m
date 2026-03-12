% Test mustBe* validator functions

% mustBeNumeric - should pass for numbers and tensors
mustBeNumeric(42);
mustBeNumeric([1 2 3]);
mustBeNumeric(true);

% mustBeNumeric - should error for strings
try
    mustBeNumeric('hello');
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'numeric')));
end

% mustBeInteger
mustBeInteger(5);
mustBeInteger([1 2 3]);
try
    mustBeInteger(1.5);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'integer')));
end

% mustBePositive
mustBePositive(1);
mustBePositive([1 2 3]);
try
    mustBePositive([-1 2]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'positive')));
end

% mustBeNonnegative
mustBeNonnegative(1);
mustBeNonnegative([1 2 3]);
try
    mustBeNonnegative([-1 2]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'nonnegative')));
end

% mustBeNonzero
mustBeNonzero(1);
mustBeNonzero([-1 2 3]);
try
    mustBeNonzero([1 0 3]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'nonzero')));
end

% mustBeFinite
mustBeFinite(42);
mustBeFinite([1 2 3]);
try
    mustBeFinite(Inf);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'finite')));
end
try
    mustBeFinite([1 NaN 3]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'finite')));
end

% mustBeNonempty
mustBeNonempty(1);
mustBeNonempty([1 2]);
mustBeNonempty('hello');
try
    mustBeNonempty([]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'nonempty')));
end

% mustBeScalarOrEmpty
mustBeScalarOrEmpty(42);
mustBeScalarOrEmpty([]);
try
    mustBeScalarOrEmpty([1 2 3]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'scalar or empty')));
end

% mustBeVector
mustBeVector(42);
mustBeVector([1 2 3]);
mustBeVector([1; 2; 3]);
try
    mustBeVector([1 2; 3 4]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'vector')));
end

% mustBeMember
mustBeMember(1, [1 2 3]);
mustBeMember(2, [1 2 3]);
try
    mustBeMember(5, [1 2 3]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'member')));
end

% mustBeInRange
mustBeInRange(5, 1, 10);
mustBeInRange([2 5 8], 1, 10);
try
    mustBeInRange(15, 1, 10);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'range')));
end

disp('SUCCESS')
