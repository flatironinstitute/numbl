classdef empty_region_holder
  properties
    datastor
    nch = 3
    hasdata = 0
  end
  properties (Dependent)
    data
  end
  methods
    function obj = empty_region_holder(n)
      obj.datastor = zeros(0, 16, n);
    end
    function data = get.data(obj)
      data = obj.datastor(:, :, 1:(obj.nch*obj.hasdata));
    end
    function obj = set.data(obj, val)
      obj.datastor(:, :, 1:(obj.nch*obj.hasdata)) = val;
    end
    function obj = resize(obj, n)
      tmp = obj.data;
      obj.datastor = zeros(0, 16, n);
      obj.data = tmp;
    end
  end
end
