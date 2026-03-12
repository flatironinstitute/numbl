classdef ColonTracker_
% Helper class for testing that colon (:) is correctly passed to subsref
% as the char ':' rather than as 0.
  methods
    function out = subsref(obj, S)
      if strcmp(S(1).type, '()')
        if isequal(S(1).subs{1}, ':')
          out = 1;  % colon received correctly
        else
          out = 0;  % colon NOT received (bug)
        end
      else
        out = builtin('subsref', obj, S);
      end
    end
  end
end
