% Regression: calling a function handle stored in a struct field with a
% comma-separated list expansion (varargin{:} / c{:}) must flatten the
% cs-list into individual arguments before invoking the handle.

outer('a', 'b', 'c');

% Direct cell expansion into a struct-field handle.
s.add = @(a, b, c) a + b + c;
c = {10, 20, 30};
assert(s.add(c{:}) == 60, 'cell expansion into struct-field handle');

% Mixed leading arg plus expansion.
s.count = @(varargin) numel(varargin);
args = {1, 2, 3, 4};
assert(s.count(args{:}) == 4, 'mixed expansion count');

disp('SUCCESS');

function outer(varargin)
  s.fn = @inner;
  assert(s.fn(s, varargin{:}) == 3, 'varargin expansion through struct-field handle');
end

function r = inner(~, varargin)
  r = numel(varargin);
end
