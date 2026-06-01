function r = stats(v, kind)
% STATS  Simple summary statistic of a vector.
%   r = stats(v, 'mean') returns the mean of v.
%   r = stats(v, 'rms')  returns the root-mean-square of v.
  switch kind
    case 'mean'
      r = sum(v) / numel(v);
    case 'rms'
      r = sqrt(sum(v .^ 2) / numel(v));
    otherwise
      error('stats: unknown kind "%s"', kind);
  end
end
