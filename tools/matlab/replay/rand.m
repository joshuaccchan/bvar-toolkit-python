function u = rand(varargin)
% rand - replay version for the tests: uniforms from the replay stream, filled in
% column-major order.
sz = replay_size(varargin{:});
u = reshape(replay_take(prod(sz)), sz);
end
