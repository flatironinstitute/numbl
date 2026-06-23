function r = plus(a, b)
a = polynom(a); b = polynom(b);
la = numel(a.c); lb = numel(b.c);
n = max(la, lb);
ca = [zeros(1, n-la), a.c];
cb = [zeros(1, n-lb), b.c];
r = polynom(ca + cb);
