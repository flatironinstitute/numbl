classdef EndTracker_
% Helper class for testing that the `end` keyword in indexing calls the
% class's overloaded end() method and that deferred ranges like 2:end are
% properly resolved before being passed to subsref.
  properties
    data
  end
  methods
    function obj = EndTracker_(d)
      if nargin > 0
        obj.data = d;
      else
        obj.data = [10 20 30 40];
      end
    end
    function e = end(obj, k, n)
      % Linear indexing: return numel; multi-dim: return size along k
      if n == 1
        e = numel(obj.data);
      else
        e = size(obj.data, k);
      end
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '()')
        % Use the indices to index into obj.data
        out = subsref(obj.data, S);
      elseif strcmp(S(1).type, '.')
        out = builtin('subsref', obj, S);
      else
        out = builtin('subsref', obj, S);
      end
    end
    function out = size(obj, dim)
      if nargin == 1
        out = size(obj.data);
      else
        out = size(obj.data, dim);
      end
    end
    function out = numel(obj)
      out = 1;
    end
  end
end
