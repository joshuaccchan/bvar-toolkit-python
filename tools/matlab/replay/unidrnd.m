function r = unidrnd(n, varargin)
% unidrnd - replay version for the tests: ceil(n*u) for the next uniforms of the
% replay stream, one per element.
if isempty(varargin)
    sz = size(n);
else
    sz = replay_size(varargin{:});
end
r = ceil(n.*reshape(replay_take(prod(sz)), sz));
end
