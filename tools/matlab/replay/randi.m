function r = randi(imax, varargin)
% randi - replay version for the tests: ceil(imax*u) for the next uniforms of the
% replay stream; imax must be a scalar.
assert(isscalar(imax), 'the replay randi takes a scalar imax');
sz = replay_size(varargin{:});
r = ceil(imax*reshape(replay_take(prod(sz)), sz));
end
