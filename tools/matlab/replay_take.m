function u = replay_take(k)
% replay_take - the next k uniforms of the replay stream, as a column. The call
% rand(stream, ...) dispatches to the RandStream method, not to replay/rand.m.
global BVT_REPLAY
u = rand(BVT_REPLAY.stream, k, 1);
BVT_REPLAY.count = BVT_REPLAY.count + k;
end
