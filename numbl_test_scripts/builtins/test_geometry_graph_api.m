% Argument checking for the geometry (delaunay, delaunayn, convhull,
% convhulln, inpolygon) and graph (graph, conncomp, laplacian, addedge,
% numnodes, numedges, degree) builtins: MATLAB's arity and type errors,
% plus degree(G, nodeIDs) and addedge weight rules.

x = [0 1 0 1]; y = [0 0 1 1];
P = [x' y'];

%% delaunay / convhull: built-ins with their own arity error ids
expect_err(@() delaunay(), 'MATLAB:delaunay:BadArgsErrId', ...
  'Invalid arguments - the input does not match the allowable argument specification.');
expect_err(@() delaunay(x, y, x, 1), 'MATLAB:delaunay:WrongInputNumArgsErrId', ...
  'Invalid number of input arguments.');
expect_err(@() two_out(@delaunay, x, y), 'MATLAB:delaunay:WrongOutputNumArgsErrId', ...
  'Invalid number of output arguments.');
expect_err(@() delaunay({1}, 2), 'MATLAB:delaunay:BadArgsErrId', ...
  'Invalid arguments - the input does not match the allowable argument specification.');
expect_err(@() delaunay([0 1 0 1i], y), 'MATLAB:delaunay:ComplexDataPointErrId', '');
expect_err(@() convhull(), 'MATLAB:convhull:BadArgsErrId', ...
  'Invalid arguments - the input does not match the allowable argument specification.');
expect_err(@() convhull(x, y, x, 1), 'MATLAB:convhull:BadArgsErrId', ...
  'Invalid arguments - the input does not match the allowable argument specification.');
expect_err(@() convhull(x, y, x, 'Simplify', true, 1), 'MATLAB:convhull:WrongInputNumArgsErrId', ...
  'Invalid number of input arguments.');
expect_err(@() three_out(@convhull, x, y), 'MATLAB:convhull:WrongOutputNumArgsErrId', ...
  'Invalid number of output arguments.');
expect_err(@() convhull([0 1 0 1i], y), 'MATLAB:convhull:ComplexDataPointErrId', '');
k = convhull(x, y, 'Simplify', true);
assert(numel(k) == 5 && k(1) == k(end));
[k, a] = convhull(P);
assert(abs(a - 1) < 1e-12);

%% delaunayn / convhulln: M-files
expect_err(@() delaunayn(), 'MATLAB:delaunayn:NotEnoughInputs', 'Not enough input arguments.');
expect_err(@() delaunayn(P, {}, 1), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() two_out(@delaunayn, P), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() delaunayn({1}), 'MATLAB:cgprechecks:NonNumericInput', ...
  'The specified data points are not in numeric array format.');
expect_err(@() delaunayn([P(:,1) P(:,2) + [0;0;0;1i]]), 'MATLAB:cgprechecks:Complex', '');
expect_err(@() convhulln(), 'MATLAB:convhulln:NotEnoughInputs', 'Needs at least 1 input.');
expect_err(@() three_out(@convhulln, P), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() convhulln([P(:,1) P(:,2) + [0;0;0;1i]]), 'MATLAB:cgprechecks:Complex', '');
assert(nargin(@delaunayn) == 2);
assert(nargin(@convhulln) == 2);

%% inpolygon
expect_err(@() inpolygon(0, 0, x), 'MATLAB:minrhs', 'Not enough input arguments.');
expect_err(@() inpolygon(0, 0, x, y, 1), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() three_out(@inpolygon, 0, 0, x, y), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() inpolygon({1}, 0, x, y), 'MATLAB:math:mustBeNumericCharOrLogical', ...
  'Invalid data type. Argument must be numeric, char, or logical.');
% char and logical query points are numeric (char codes, 0/1)
assert(isequal(inpolygon('a', 0, [0 1 0], [0 0 1]), false));
assert(isequal(inpolygon(true, 0, [0 1 0], [0 0 1]), true));
assert(nargin(@inpolygon) == 4);

%% graph constructor
expect_err(@() graph(1, 2, 1, 3, 'omitselfloops', 1), 'MATLAB:maxrhs', 'Too many input arguments.');
expect_err(@() two_out(@graph, [1 2], [2 3]), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() graph([1 1i], [2 3]), 'MATLAB:graphfun:graphbuiltin:InvalidSRC', ...
  'Source must be a numeric array of positive integer node indices.');
expect_err(@() graph([1 2], [2 1i]), 'MATLAB:graphfun:graphbuiltin:InvalidTARG', ...
  'Target must be a numeric array of positive integer node indices.');
expect_err(@() graph([1 2], [2 3], [1 1i]), 'MATLAB:graphfun:graph:InvalidWeights', ...
  'Graph edge weights must be double or single, real, and not sparse.');
expect_err(@() graph([1 2], [2 3], 1i), 'MATLAB:graphfun:graph:InvalidWeights', ...
  'Graph edge weights must be double or single, real, and not sparse.');
expect_err(@() graph([0 1i; 1i 0]), 'MATLAB:graphfun:graphbuiltin:InvalidAdjacency', ...
  'Invalid data type. Adjacency matrix must be a real double, single, or logical square matrix.');
G5 = graph([1 2], [2 3], [1 2], 5, 'omitselfloops');
assert(numnodes(G5) == 5);

%% graph methods: a non-graph argument is an undefined function in MATLAB
G = graph([1 2 3 3], [2 3 1 4]);
expect_err(@() conncomp(1), 'MATLAB:UndefinedFunction', ...
  'Undefined function ''conncomp'' for input arguments of type ''double''.');
expect_err(@() numnodes('a'), 'MATLAB:UndefinedFunction', ...
  'Undefined function ''numnodes'' for input arguments of type ''char''.');
expect_err(@() numedges({1}), 'MATLAB:UndefinedFunction', ...
  'Undefined function ''numedges'' for input arguments of type ''cell''.');
expect_err(@() degree(true), 'MATLAB:UndefinedFunction', ...
  'Undefined function ''degree'' for input arguments of type ''logical''.');
expect_err(@() laplacian(struct()), 'MATLAB:UndefinedFunction', ...
  'Undefined function ''laplacian'' for input arguments of type ''struct''.');
expect_err(@() laplacian(G, 1), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() numnodes(G, 1), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() numedges(G, 1), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() degree(G, 1, 2), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() addedge(G), 'MATLAB:minrhs', 'Not enough input arguments.');
expect_err(@() addedge(G, 1), 'MATLAB:minrhs', 'Not enough input arguments.');
expect_err(@() addedge(G, 1, 2, 3, 4), 'MATLAB:TooManyInputs', 'Too many input arguments.');
expect_err(@() two_out(@laplacian, G), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() two_out(@numnodes, G), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() two_out(@degree, G), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
expect_err(@() three_out(@conncomp, G), 'MATLAB:TooManyOutputs', 'Too many output arguments.');

%% conncomp options
assert(isequal(conncomp(G, 'OutputForm', 'vector'), [1 1 1 1]));

%% degree(G, nodeIDs)
assert(isequal(degree(G, [1 3]), [2 3]));
assert(isequal(degree(G, [1; 3]), [2; 3]));
assert(isequal(degree(G, 3), 3));
assert(isequal(degree(G, ones(2)), [2 2; 2 2]));
expect_err(@() degree(G, 5), 'MATLAB:graphfun:graph:InvalidNodeID', ...
  'Numeric node IDs must be positive integers not greater than the number of nodes in the graph (4).');
expect_err(@() degree(G, 1.5), 'MATLAB:graphfun:graph:InvalidNodeID', ...
  'Numeric node IDs must be positive integers not greater than the number of nodes in the graph (4).');

%% addedge node IDs and weights
expect_err(@() addedge(G, 1i, 2), 'MATLAB:graphfun:addedge:InvalidNodeID', ...
  'Numeric node IDs must be positive integers.');
expect_err(@() addedge(G, 0, 2), 'MATLAB:graphfun:addedge:InvalidNodeID', ...
  'Numeric node IDs must be positive integers.');
expect_err(@() addedge(G, 1, 4, 2), 'MATLAB:graphfun:addedge:NoWeights', ...
  'Unable to add weighted edges to an unweighted graph.');
W = graph([1 2], [2 3], [5 6]);
W2 = addedge(W, 1, 3, 7);
assert(numedges(W2) == 3);

disp('SUCCESS')

function expect_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('expected id %s, got %s (%s)', id, e.identifier, e.message));
    if ~isempty(msg)
      assert(strcmp(e.message, msg), sprintf('expected message "%s", got "%s"', msg, e.message));
    end
    return;
  end
  error('expected error %s', id);
end

function two_out(f, varargin)
  [a, b] = f(varargin{:}); %#ok<ASGLU>
end

function three_out(f, varargin)
  [a, b, c] = f(varargin{:}); %#ok<ASGLU>
end
