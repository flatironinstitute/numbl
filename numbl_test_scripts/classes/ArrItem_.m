classdef ArrItem_
  % Helper for object_array_member_and_end.m: a simple value class so we can
  % build object arrays.
  properties
    v = 0
  end
  methods
    function obj = ArrItem_(x)
      if nargin > 0
        obj.v = x;
      end
    end
  end
end
