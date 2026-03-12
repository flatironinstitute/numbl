classdef PrefHolder_
  methods (Static = true)
    function varargout = getDefaults(varargin)
      persistent defaults
      if isempty(defaults)
        defaults = struct();
        defaults.alpha = 10;
        defaults.beta = 20;
      end
      if strcmp(varargin{1}, 'get')
        varargout{1} = defaults;
      elseif strcmp(varargin{1}, 'set')
        defaults = varargin{2};
      end
    end
  end
end
