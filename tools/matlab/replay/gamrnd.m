function r = gamrnd(a, b, varargin)
% gamrnd - replay version for the tests: gamma draws with shape a and scale b by the
% inverse distribution function of the next uniforms of the replay stream.
if isempty(varargin)
    sz = size(a.*b);
else
    sz = replay_size(varargin{:});
end
u = reshape(replay_take(prod(sz)), sz);
r = gammaincinv(u, a).*b;
end
