function out = value(obj, flag)
%VALUE  Method file sharing name with 'value' property.
if nargin == 2 && strcmp(flag, 'double')
    out = obj.value * 2;
else
    out = obj.value;
end
end
