function z = randn(varargin)
% randn - replay version for the tests: the inverse normal distribution function of
% the next uniforms of the replay stream.
sz = replay_size(varargin{:});
z = reshape(-sqrt(2)*erfcinv(2*replay_take(prod(sz))), sz);
end
