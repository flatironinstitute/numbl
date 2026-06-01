function obj = differentName(obj)
% The file is named bump.m but the function inside is differentName.
% MATLAB dispatches external methods by FILE name, using the file's primary
% function regardless of its declared name (a tolerated mismatch, as in
% chebfun's @chebfun/resetPointValues.m which declares clearPointValues).
obj.value = obj.value + 100;
end
