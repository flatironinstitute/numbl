classdef SubsrefInternal_
  % Test that obj(k) and obj(k).prop inside class methods use built-in
  % array indexing, NOT the overloaded subsref.
  properties
    myFlag
  end
  methods
    function obj = SubsrefInternal_(val)
      obj.myFlag = val;
    end
    function varargout = subsref(obj, S)
      % Overloaded subsref — intercepts external () and . access.
      % Inside class methods this should NOT be called for obj(k).prop.
      if strcmp(S(1).type, '()')
        % External paren indexing: double the value at the requested point
        varargout{1} = obj.myFlag * 2;
      else
        % For dot access (method calls, property access), use builtin
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      end
    end
    function out = getFlag(obj)
      % Inside class method: obj(1).myFlag must use built-in behavior
      % (return myFlag directly), not overloaded subsref (which doubles).
      out = obj(1).myFlag;
    end
    function out = getSelf(obj)
      % obj(1) inside class method should return obj itself
      out = obj(1);
    end
  end
end
