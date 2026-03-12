classdef BaseAnimal_
  properties
    Name
    Legs
  end
  methods
    function obj = BaseAnimal_(name, legs)
      obj.Name = name;
      obj.Legs = legs;
    end
    function r = describe(obj)
      r = obj.Legs;
    end
  end
end
