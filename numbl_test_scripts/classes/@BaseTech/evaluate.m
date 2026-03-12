function y = evaluate(x, coeffs)
%EVALUATE  Static method on BaseTech: compute coeffs(1) + coeffs(2)*x
%   Takes (x, coeffs), NOT (obj, x, coeffs).
    y = coeffs(1) + coeffs(2) * x;
end
