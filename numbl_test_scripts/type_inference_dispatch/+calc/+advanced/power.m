function r = power(base, exp)
    % Nested package function calling parent package function
    if exp == 0
        r = 1;
    elseif exp == 1
        r = base;
    else
        r = base * calc.advanced.power(base, exp - 1);
    end
end
