function replay_start(seed)
% replay_start - route rand, randn, gamrnd, betarnd, unidrnd, randi, randperm and iwishrnd
% to the uniforms of RandStream('mt19937ar', 'Seed', seed) until replay_stop, by putting
% the functions in replay/ first on the path; gigrnd returns the mode of its distribution, and
% fminunc is the Nelder-Mead search of the Python port. The Python twin is bvar_toolkit.Replay.
global BVT_REPLAY
BVT_REPLAY.stream = RandStream('mt19937ar', 'Seed', seed);
BVT_REPLAY.count = 0;
w = warning('off', 'MATLAB:dispatcher:nameConflict');
addpath(fullfile(fileparts(mfilename('fullpath')), 'replay'), '-begin');
warning(w);
end
