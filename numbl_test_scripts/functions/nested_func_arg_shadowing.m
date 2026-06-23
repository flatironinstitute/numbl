% Test: a nested function's own formal in/out arguments are LOCAL to it and
% must not be redirected to a same-named variable in the parent workspace.
% (This is the spatialmath-matlab UnitQuaternion.tr2q pattern: a nested
% `function s = sign(x)` whose output `s` collides with the parent's `s`.)
% Only NON-argument variables are shared with the parent.
% All expected values verified against MATLAB R2025b.

% Output arg of nested fn shadows parent var: parent's s survives the call.
assert(out_shadow() == 0.84167, 'nested output arg must not clobber parent var');

% Input arg of nested fn shadows parent var: parent's x survives the call.
assert(in_shadow() == 5, 'nested input arg must not clobber parent var');

% Sanity: legitimate sharing still works (a non-arg variable IS shared).
assert(shared_still_works() == 42, 'non-arg parent var should still be shared');

disp('SUCCESS')

function r = out_shadow()
    function s = sgnfn(x)   % output `s` shadows the parent's `s`
        if x >= 0
            s = 1;
        else
            s = -1;
        end
    end
    s = 0.84167;
    g = sgnfn(5);           %#ok<NASGU>  must not overwrite parent `s`
    r = s;
end

function r = in_shadow()
    function y = addone(x)  % input `x` shadows the parent's `x`
        y = x + 1;
    end
    x = 5;
    z = addone(100);        %#ok<NASGU>  must not overwrite parent `x`
    r = x;
end

function r = shared_still_works()
    function bump()         % no formal args; `acc` is genuinely shared
        acc = acc + 2;
    end
    acc = 40;
    bump();
    r = acc;                % should be 42
end
