function out = domain(obj, flag)
%DOMAIN  Method file sharing name with 'domain' property.
%   When called via function syntax domain(obj), returns the domain property.
%   When called with flag 'ends', returns first and last elements.
if nargin == 2 && strcmp(flag, 'ends')
    d = obj.domain;
    out = [d(1) d(end)];
else
    out = obj.domain;
end
end
