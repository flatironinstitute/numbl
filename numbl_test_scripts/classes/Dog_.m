classdef Dog_ < BaseAnimal_
  properties
    Breed
  end
  methods
    function obj = Dog_(breed)
      obj = obj@BaseAnimal_('dog', 4);
      obj.Breed = breed;
    end
    function r = describe(obj)
      r = describe@BaseAnimal_(obj) * 10;
    end
  end
end
