function Y = normalize(X, method, p)
    % normalize(X)            z-score along the first non-singleton dimension
    % normalize(X, 'zscore')  same as above
    % normalize(X, 'norm')    divide by the 2-norm along that dimension
    % normalize(X, 'norm', p) divide by the p-norm (p = Inf -> max abs)
    if nargin < 2
        method = 'zscore';
    end
    if size(X, 1) == 1
        dim = 2;
    else
        dim = 1;
    end
    switch lower(method)
        case 'zscore'
            mu = mean(X, dim);
            sigma = std(X, 0, dim);
            Y = (X - mu) ./ sigma;
        case 'norm'
            if nargin < 3
                p = 2;
            end
            if isinf(p)
                n = max(abs(X), [], dim);
            else
                n = sum(abs(X) .^ p, dim) .^ (1 / p);
            end
            Y = X ./ n;
        otherwise
            error('normalize: unsupported method ''%s''', method);
    end
end
