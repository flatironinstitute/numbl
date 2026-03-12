classdef PropDefaultTest_
  properties
    x    % no explicit default - should be []
    y = 5  % explicit default - should be 5
  end
  methods
    function obj = PropDefaultTest_()
    end
  end
end
