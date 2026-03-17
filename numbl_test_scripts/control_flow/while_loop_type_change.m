% Test that while loop condition is evaluated correctly when variables
% change type inside the loop body (e.g. from Number to Unknown via multi-assign).

function while_loop_type_change()
g1 = 17;
isHappy = 0;
failure = 0;
iter = 0;
while ~isHappy && ~failure
    iter = iter + 1;
    [g1, resolved] = step(g1);
    isHappy = all(resolved);
    if g1 > 200
        failure = 1;
    end
end
assert(iter > 1, 'while loop should have run more than once');
assert(isHappy == 1, 'should have reached happy state');
assert(g1 > 80, 'g1 should have grown past 80');
disp('SUCCESS');
end

function [g1, res] = step(a)
g1 = round(1.5 * a);
res = [g1 > 80];
end
