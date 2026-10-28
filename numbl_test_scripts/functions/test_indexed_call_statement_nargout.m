% A call through an indexed function handle (c{i}(), s.f(), h(i)) used as a
% statement requests no outputs, so it may call a function that has none.
% ans is set only when a value is returned.

global calls
calls = {};

c = {@no_outputs, @() no_outputs(), @counted_outputs};
c{1}();
c{1}(5);
c{2}();
assert(isequal(calls, {0, 0, 0}));
assert(~exist('ans', 'var'));

% A function with outputs still sets ans; it sees nargout 0.
c{3}();
assert(isequal(calls{end}, 0));
assert(isequal(ans, 1)); %#ok<NOANS>

% Requested outputs are passed through.
[a, b] = c{3}();
assert(a == 1 && b == 2);
assert(isequal(calls{end}, 2));

s.f = @no_outputs;
s.f();
assert(numel(calls) == 6);

% Void builtins stored in a cell.
f = [tempname '.txt'];
fid = fopen(f, 'w');
fprintf(fid, 'abc');
fclose(fid);
fid = fopen(f, 'r');
fread(fid, 2);
ops = {@frewind};
ops{1}(fid);
assert(ftell(fid) == 0);
try
  x = frewind(fid); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Too many output arguments.'));
end
fclose(fid);
delete(f);

disp('SUCCESS');

function no_outputs(varargin)
  global calls
  calls{end + 1} = nargout;
end

function varargout = counted_outputs(varargin)
  global calls
  calls{end + 1} = nargout;
  for k = 1:max(nargout, 1)
    varargout{k} = k;
  end
end
