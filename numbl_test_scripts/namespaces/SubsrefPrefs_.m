classdef SubsrefPrefs_
  properties
    store
  end
  methods
    function obj = SubsrefPrefs_()
      inner = struct;
      inner.epsilon = 0.001;
      inner.maxiter = 500;
      s = struct;
      s.alpha = 10;
      s.beta = 20;
      s.techPrefs = inner;
      obj.store = s;
    end
    function out = subsref(obj, S)
      out = obj.store;
      for k = 1:length(S)
        if strcmp(S(k).type, '.')
          out = out.(S(k).subs);
        end
      end
    end
  end
end
