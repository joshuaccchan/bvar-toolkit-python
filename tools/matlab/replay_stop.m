function count = replay_stop()
% replay_stop - restore the built-in random number functions and return the number
% of uniforms drawn since replay_start.
global BVT_REPLAY
w = warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(fullfile(fileparts(mfilename('fullpath')), 'replay'));
warning(w);
count = BVT_REPLAY.count;
end
