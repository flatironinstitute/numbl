classdef ParenAsgn_
% Helper class for testing that paren-indexed assignment (obj(idx) = val)
% routes through the user-defined subsasgn method.
  properties
    data = 0
    call_count = 0
  end
  methods
    function obj = ParenAsgn_(val)
      if nargin > 0
        obj.data = val;
      end
    end
    function obj = subsasgn(obj, S, val)
      if strcmp(S(1).type, '()')
        % Record that subsasgn was called
        obj.call_count = obj.call_count + 1;
        % For empty index, just return obj unchanged (no-op)
        if isempty(S(1).subs{end})
          return;
        end
        % For non-empty index, store the value
        obj.data = val;
      else
        obj = builtin('subsasgn', obj, S, val);
      end
    end
    function result = assignSameClassInside(obj, other)
      % Inside a class method: obj(1) = other should bypass subsasgn
      % and do direct object replacement (MATLAB classdef behavior).
      obj(1) = other;
      result = obj;
    end
  end
end
