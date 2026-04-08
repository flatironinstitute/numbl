classdef DepPropTensor_
  properties
    stor
  end
  properties (Dependent)
    view        % returns a slice of stor as a 2D tensor
    view3d      % returns stor itself, a 3D tensor
  end
  methods
    function obj = DepPropTensor_(data3d)
      obj.stor = data3d;
    end
    function v = get.view(obj)
      v = obj.stor(:, :, 1);
    end
    function v = get.view3d(obj)
      v = obj.stor;
    end
  end
end
