% Tests for the undirected graph object: graph, conncomp, laplacian,
% addedge, numnodes, numedges, degree. Runs in both numbl and MATLAB.

%% graph from an adjacency matrix (4-cycle)
A = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0];
G = graph(A);
assert(strcmp(class(G), 'graph'));
assert(numnodes(G) == 4);
assert(numedges(G) == 4);
assert(isequal(conncomp(G), [1 1 1 1]));
assert(isequal(degree(G), [2; 2; 2; 2]));
Lexp = [2 -1 0 -1; -1 2 -1 0; 0 -1 2 -1; -1 0 -1 2];
assert(isequal(full(laplacian(G)), Lexp));

%% 'omitselfloops' drops the diagonal (5-point stencil -> same 4-cycle)
A2 = [4 -1 0 -1; -1 4 -1 0; 0 -1 4 -1; -1 0 -1 4];
G2 = graph(A2, 'omitselfloops');
assert(numedges(G2) == 4);
assert(isequal(full(laplacian(G2)), Lexp));

%% laplacian ignores edge weights (uses binary adjacency + degree count)
Aw = [0 2 0; 2 0 2; 0 2 0];
Gw = graph(Aw);
assert(isequal(full(laplacian(Gw)), [1 -1 0; -1 2 -1; 0 -1 1]));

%% disconnected graph -> components labeled by smallest node index
B = zeros(5);
B(2,3) = 1; B(3,2) = 1; B(3,4) = 1; B(4,3) = 1;
H = graph(B);
[bins, sz] = conncomp(H);
assert(isequal(bins, [1 2 2 2 3]));
assert(isequal(sz, [1 3 1]));

%% addedge connects two components of an unweighted graph
D = graph(logical([0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]));
assert(isequal(conncomp(D), [1 1 2 2]));
D2 = addedge(D, 1, 3);
assert(isequal(conncomp(D2), [1 1 1 1]));
assert(numedges(D2) == 3);

%% edge-list constructors
Gt = graph([1 2 3], [2 3 1]);          % triangle
assert(numnodes(Gt) == 3);
assert(numedges(Gt) == 3);
assert(isequal(conncomp(Gt), [1 1 1]));
Gp = graph([1 2], [2 3], [10 20]);     % weighted path
assert(numedges(Gp) == 2);
assert(isequal(full(laplacian(Gp)), [1 -1 0; -1 2 -1; 0 -1 1]));

%% laplacian is sparse and feeds eigs (Fiedler value of a path graph)
P = graph(1:5, 2:6);                    % path 1-2-...-6
L = laplacian(P);
assert(issparse(L));
[V, Dl] = eigs(L, 2, 'SA');
d = sort(diag(Dl));
assert(abs(d(1)) < 1e-9);               % connected -> smallest eigenvalue 0
assert(d(2) > 1e-6);

disp('SUCCESS')
