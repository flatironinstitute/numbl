classdef PropValid_
  % Property size/class/validator declarations, applied to defaults and
  % to every assignment.
  properties
    a (1,1) double {mustBePositive} = 1
    b (1,:) char = 'x'
    c double
    d (1,3) = 0
    e (1,1) double
    h (1,1) logical
    k {mustBeNumeric}
    m (2,2) double
    r (1,:) double
    s (1,2)
    u (1,1) char
    w (1,1) string
    z (1,:) cell
    color {mustBeMember(color, {'red', 'blue'})} = 'red'
    count (1,1) double {mustBeInteger, mustBeNonnegative} = 0
  end
  methods
    function obj = bump(obj)
      obj.count = obj.count + 1;
    end
  end
end
