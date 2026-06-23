% Spherical conformal map of a genus-0 closed surface (Choi/Lam/Lui FLASH).
% This is a regression test for numbl's sparse + complex linear algebra:
% it builds a sparse cotangent Laplacian, solves M\complex(c,d) (real sparse
% matrix, complex RHS), multiplies sparse * dense, edits a sparse matrix
% (find -> sparse -> add/subtract, row/col assignment), and solves a complex
% sparse system A\b. The whole pipeline must match MATLAB bit-for-bit.
%
% The mesh is built deterministically (a Fibonacci-sphere lattice stretched
% into an ellipsoid) so the result is identical in numbl and MATLAB without
% depending on the RNG. The reference algorithm is by Gary P.-T. Choi
% (https://github.com/garyptchoi/spherical-conformal-map), MIT licensed.

% --- Deterministic genus-0 closed triangle mesh: Fibonacci sphere -> hull,
%     then stretched into an ellipsoid so the conformal map does real work. ---
n = 300;
idx = (0:n-1)';
phi = acos(1 - 2*(idx+0.5)/n);
golden = pi*(1+sqrt(5));
theta = golden*idx;
p = [sin(phi).*cos(theta), sin(phi).*sin(theta), cos(phi)];
f = convhull(p(:,1), p(:,2), p(:,3));
v = p .* [2.0, 1.0, 0.6];

% Euler characteristic of a genus-0 closed mesh: V - E + F = 2.
nv = size(v,1); nf = size(f,1);
assert(nv - 3*nf/2 + nf == 2, 'constructed mesh is not genus-0 closed');

% --- Conformal map to the unit sphere ---
map = spherical_conformal_map(v, f);

assert(isequal(size(map), size(v)), 'map has wrong size');
assert(all(isfinite(map(:))), 'map contains non-finite values');

% Every mapped vertex must lie on the unit sphere.
r = sqrt(sum(map.^2, 2));
assert(max(abs(r - 1)) < 1e-3, ...
    sprintf('mapped vertices are not on the unit sphere (max |r-1| = %g)', max(abs(r-1))));

% Angle distortion must be small for a conformal map.
fa = @(a,b) acos(sum(a.*b,2) ./ (sqrt(sum(a.^2,2)).*sqrt(sum(b.^2,2))));
ang = @(V) [fa(V(f(:,2),:)-V(f(:,1),:), V(f(:,3),:)-V(f(:,1),:)); ...
            fa(V(f(:,1),:)-V(f(:,2),:), V(f(:,3),:)-V(f(:,2),:)); ...
            fa(V(f(:,1),:)-V(f(:,3),:), V(f(:,2),:)-V(f(:,3),:))];
distortion = abs(ang(map) - ang(v)) * 180/pi;
assert(median(distortion) < 8, ...
    sprintf('angle distortion too large (median %g deg)', median(distortion)));

disp('SUCCESS')

% ─────────────────────────────────────────────────────────────────────────
% Reference implementation (local functions), faithful to the FLASH paper.

function map = spherical_conformal_map(v,f)
    if length(v)-3*length(f)/2+length(f) ~= 2
        error('The mesh is not a genus-0 closed surface.');
    end

    % Most regular triangle as the "big triangle"
    temp = v(reshape(f',1,length(f)*3),1:3);
    e1 = sqrt(sum((temp(2:3:end,1:3) - temp(3:3:end,1:3))'.^2))';
    e2 = sqrt(sum((temp(1:3:end,1:3) - temp(3:3:end,1:3))'.^2))';
    e3 = sqrt(sum((temp(1:3:end,1:3) - temp(2:3:end,1:3))'.^2))';
    regularity = abs(e1./(e1+e2+e3)-1/3)+abs(e2./(e1+e2+e3)-1/3)+abs(e3./(e1+e2+e3)-1/3);
    [~,bigtri] = min(regularity);

    % North pole step: harmonic map by solving the Laplace equation
    nv = size(v,1);
    M = cotangent_laplacian(v,f);
    p1 = f(bigtri,1); p2 = f(bigtri,2); p3 = f(bigtri,3);
    fixed = [p1,p2,p3];
    [mrow,mcol,mval] = find(M(fixed,:));
    M = M - sparse(fixed(mrow),mcol,mval,nv,nv) + sparse(fixed,fixed,[1,1,1],nv,nv);

    x1 = 0; y1 = 0; x2 = 1; y2 = 0;
    a = v(p2,1:3) - v(p1,1:3);
    b = v(p3,1:3) - v(p1,1:3);
    sin1 = (norm(cross(a,b),2))/(norm(a,2)*norm(b,2));
    ori_h = norm(b,2)*sin1;
    ratio = norm([x1-x2,y1-y2],2)/norm(a,2);
    y3 = ori_h*ratio;
    x3 = sqrt(norm(b,2)^2*ratio^2-y3^2);

    c = zeros(nv,1); c(p1) = x1; c(p2) = x2; c(p3) = x3;
    d = zeros(nv,1); d(p1) = y1; d(p2) = y2; d(p3) = y3;
    z = M \ complex(c,d);
    z = z-mean(z);

    S = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];

    % Optimal big-triangle size
    w = complex(S(:,1)./(1+S(:,3)), S(:,2)./(1+S(:,3)));
    [~, index] = sort(abs(z(f(:,1)))+abs(z(f(:,2)))+abs(z(f(:,3))));
    inner = index(1);
    if inner == bigtri; inner = index(2); end
    NorthTriSide = (abs(z(f(bigtri,1))-z(f(bigtri,2))) + abs(z(f(bigtri,2))-z(f(bigtri,3))) + abs(z(f(bigtri,3))-z(f(bigtri,1))))/3;
    SouthTriSide = (abs(w(f(inner,1))-w(f(inner,2))) + abs(w(f(inner,2))-w(f(inner,3))) + abs(w(f(inner,3))-w(f(inner,1))))/3;
    z = z*(sqrt(NorthTriSide*SouthTriSide))/(NorthTriSide);
    S = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), (-1+abs(z).^2)./(1+abs(z).^2)];

    % South pole step
    [~,I] = sort(S(:,3));
    fixnum = max(round(length(v)/10),3);
    fixed = I(1:min(length(v),fixnum));
    P = [S(:,1)./(1+S(:,3)), S(:,2)./(1+S(:,3))];
    mu = beltrami_coefficient(P, f, v);
    map = linear_beltrami_solver(P,f,mu,fixed,P(fixed,:));

    z = complex(map(:,1),map(:,2));
    map = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), -(abs(z).^2-1)./(1+abs(z).^2)];
end

function L = cotangent_laplacian(v,f)
    nv = length(v);
    f1 = f(:,1); f2 = f(:,2); f3 = f(:,3);
    l1 = sqrt(sum((v(f2,:) - v(f3,:)).^2,2));
    l2 = sqrt(sum((v(f3,:) - v(f1,:)).^2,2));
    l3 = sqrt(sum((v(f1,:) - v(f2,:)).^2,2));
    s = (l1 + l2 + l3)*0.5;
    area = sqrt( s.*(s-l1).*(s-l2).*(s-l3));
    cot12 = (l1.^2 + l2.^2 - l3.^2)./area/2;
    cot23 = (l2.^2 + l3.^2 - l1.^2)./area/2;
    cot31 = (l1.^2 + l3.^2 - l2.^2)./area/2;
    diag1 = -cot12-cot31; diag2 = -cot12-cot23; diag3 = -cot31-cot23;
    II = [f1; f2; f2; f3; f3; f1; f1; f2; f3];
    JJ = [f2; f1; f3; f2; f1; f3; f1; f2; f3];
    V = [cot12; cot12; cot23; cot23; cot31; cot31; diag1; diag2; diag3];
    L = sparse(II,JJ,V,nv,nv);
end

function mu = beltrami_coefficient(v, f, map)
    nf = length(f);
    Mi = reshape([1:nf;1:nf;1:nf], [1,3*nf]);
    Mj = reshape(f', [1,3*nf]);
    e1 = v(f(:,3),1:2) - v(f(:,2),1:2);
    e2 = v(f(:,1),1:2) - v(f(:,3),1:2);
    e3 = v(f(:,2),1:2) - v(f(:,1),1:2);
    area = (-e2(:,1).*e1(:,2) + e1(:,1).*e2(:,2))'/2;
    area = [area;area;area];
    Mx = reshape([e1(:,2),e2(:,2),e3(:,2)]'./area /2 , [1, 3*nf]);
    My = -reshape([e1(:,1),e2(:,1),e3(:,1)]'./area /2 , [1, 3*nf]);
    Dx = sparse(Mi,Mj,Mx);
    Dy = sparse(Mi,Mj,My);
    dXdu = Dx*map(:,1); dXdv = Dy*map(:,1);
    dYdu = Dx*map(:,2); dYdv = Dy*map(:,2);
    dZdu = Dx*map(:,3); dZdv = Dy*map(:,3);
    E = dXdu.^2 + dYdu.^2 + dZdu.^2;
    G = dXdv.^2 + dYdv.^2 + dZdv.^2;
    F = dXdu.*dXdv + dYdu.*dYdv + dZdu.*dZdv;
    mu = (E - G + 2 * 1i * F) ./ (E + G + 2*sqrt(E.*G - F.^2));
end

function map = linear_beltrami_solver(v,f,mu,landmark,target)
    af = (1-2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
    bf = -2*imag(mu)./(1.0-abs(mu).^2);
    gf = (1+2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
    f0 = f(:,1); f1 = f(:,2); f2 = f(:,3);
    uxv0 = v(f1,2) - v(f2,2); uyv0 = v(f2,1) - v(f1,1);
    uxv1 = v(f2,2) - v(f0,2); uyv1 = v(f0,1) - v(f2,1);
    uxv2 = v(f0,2) - v(f1,2); uyv2 = v(f1,1) - v(f0,1);
    l = [sqrt(sum(uxv0.^2 + uyv0.^2,2)), sqrt(sum(uxv1.^2 + uyv1.^2,2)), sqrt(sum(uxv2.^2 + uyv2.^2,2))];
    s = sum(l,2)*0.5;
    area = sqrt(s.*(s-l(:,1)).*(s-l(:,2)).*(s-l(:,3)));
    v00 = (af.*uxv0.*uxv0 + 2*bf.*uxv0.*uyv0 + gf.*uyv0.*uyv0)./area;
    v11 = (af.*uxv1.*uxv1 + 2*bf.*uxv1.*uyv1 + gf.*uyv1.*uyv1)./area;
    v22 = (af.*uxv2.*uxv2 + 2*bf.*uxv2.*uyv2 + gf.*uyv2.*uyv2)./area;
    v01 = (af.*uxv1.*uxv0 + bf.*uxv1.*uyv0 + bf.*uxv0.*uyv1 + gf.*uyv1.*uyv0)./area;
    v12 = (af.*uxv2.*uxv1 + bf.*uxv2.*uyv1 + bf.*uxv1.*uyv2 + gf.*uyv2.*uyv1)./area;
    v20 = (af.*uxv0.*uxv2 + bf.*uxv0.*uyv2 + bf.*uxv2.*uyv0 + gf.*uyv0.*uyv2)./area;
    I = [f0;f1;f2;f0;f1;f1;f2;f2;f0];
    J = [f0;f1;f2;f1;f0;f2;f1;f0;f2];
    V = [v00;v11;v22;v01;v01;v12;v12;v20;v20]/2;
    A = sparse(I,J,-V);
    targetc = target(:,1) + 1i*target(:,2);
    b = -A(:,landmark)*targetc;
    b(landmark) = targetc;
    A(landmark,:) = 0; A(:,landmark) = 0;
    A = A + sparse(landmark,landmark,ones(length(landmark),1), size(A,1), size(A,2));
    map = A\b;
    map = [real(map),imag(map)];
end
