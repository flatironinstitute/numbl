% Test usejava builtin

% Common behavior: returns a logical scalar for any feature string.
for feat = {'awt', 'jvm', 'swing', 'desktop'}
    tf = usejava(feat{1});
    assert(islogical(tf));
    assert(isscalar(tf));
end

% numbl always returns false; MATLAB depends on the runtime.
if exist('isnumbl', 'builtin') == 5
    assert(~usejava('awt'));
    assert(~usejava('jvm'));
    assert(~usejava('swing'));
    assert(~usejava('desktop'));
end

disp('SUCCESS');
