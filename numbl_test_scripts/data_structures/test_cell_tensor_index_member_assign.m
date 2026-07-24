% Regression: `c{k}.field = value` where k is a 1x1 tensor index (e.g. from
% find) must extract the single cell element as the member-assign base rather
% than leave it wrapped as a one-element comma-list. Mirrors pulseq's
% check_g{channelNum}.start = [...] with channelNum = find(strcmp(...)),
% which threw "Expected a runtime value, got [tensor]".

c = cell(1, 3);
ch = find(strcmp('y', {'x', 'y', 'z'}));   % 1x1 double, value 2
assert(ch == 2);

c{ch}.start = [1 2];
c{ch}.stop = [3 4];
assert(isequal(c{2}.start, [1 2]));
assert(isequal(c{2}.stop, [3 4]));

% Deeper member chain with a tensor cell index
d = cell(1, 2);
k = find([1 0]);
d{k}.a.b = 7;
assert(d{1}.a.b == 7);

% Growing a numeric field through a tensor cell index
e = cell(1, 2);
j = find([0 1]);
e{j}.vals(3) = 5;
assert(isequal(e{2}.vals, [0 0 5]));

% Plain-literal index still works (no regression)
f = cell(1, 2);
f{2}.x = 9;
assert(f{2}.x == 9);

disp('SUCCESS');
