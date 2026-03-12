function r = sum_powers(base, max_exp)
    % Nested package function calling sibling and parent package functions
    r = 0;
    for e = 0:max_exp
        r = calc.add(r, calc.advanced.power(base, e));
    end
end
