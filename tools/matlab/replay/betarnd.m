function r = betarnd(a, b, varargin)
% betarnd - replay version for the tests: beta draws by the inverse distribution
% function of the next uniforms of the replay stream.
if isempty(varargin)
    sz = size(a.*b);
else
    sz = replay_size(varargin{:});
end
u = reshape(replay_take(prod(sz)), sz);
r = betaincinv(u, a, b);
end
