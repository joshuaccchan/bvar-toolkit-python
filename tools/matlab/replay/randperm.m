function p = randperm(k)
% randperm - replay version for the tests: the order that sorts the next k
% uniforms of the replay stream.
[~, p] = sort(replay_take(k)');
end
