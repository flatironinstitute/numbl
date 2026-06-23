classdef ImportMethodWild_
  methods
    function y = compute(~, x)
      % A wildcard import inside a classdef method body must resolve a bare
      % package-function call.
      import impwild.*
      y = leaf(x) + 2;
    end
  end
end
