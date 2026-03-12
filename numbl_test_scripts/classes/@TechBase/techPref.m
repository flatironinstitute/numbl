function outPref = techPref(inPref)
%TECHPREF  Static method in external file.
%   Mirrors chebfun pattern: chebtech.techPref(pref).

outPref.alpha = 10;
outPref.beta = 20;

if nargin == 1
    fns = fieldnames(inPref);
    for k = 1:length(fns)
        outPref.(fns{k}) = inPref.(fns{k});
    end
end

end
