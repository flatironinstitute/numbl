classdef SubsrefLocalFn_
  % Test that file-local subfunctions also use built-in indexing
  % for instances of the same class, not overloaded subsref.
  properties
    myFlag
  end
  methods
    function obj = SubsrefLocalFn_(val)
      obj.myFlag = val;
    end
    function varargout = subsref(obj, S)
      if strcmp(S(1).type, '()')
        varargout{1} = obj.myFlag * 2;
      else
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      end
    end
    function out = getViaHelper(obj)
      % Delegates to a file-local subfunction
      out = helperGetFlag(obj);
    end
  end
end

function out = helperGetFlag(f)
  % File-local subfunction: f(1).myFlag should use built-in indexing
  out = f(1).myFlag;
end
