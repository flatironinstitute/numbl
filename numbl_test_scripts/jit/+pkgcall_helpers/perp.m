function n = perp(tau)
% Mirrors chnk.perp's body: return the in-plane perpendicular of a 2xN
% direction tensor. The reshape is a no-op for 2D input but keeps the
% same code shape as chunkie (so we exercise the same JIT path).
n = [tau(2,:); -tau(1,:)];
n = reshape(n, size(tau));
end
