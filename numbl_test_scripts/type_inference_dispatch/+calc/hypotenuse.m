function r = hypotenuse(a, b)
    % Package function calling two other package functions
    r = sqrt(calc.add(calc.square(a), calc.square(b)));
end
