function sz = replay_size(varargin)
% replay_size - the array size requested by the arguments of rand(...), randn(...)
% or the size arguments of gamrnd(a, b, ...); class names such as 'double' are ignored.
args = varargin(cellfun(@isnumeric, varargin));
if isempty(args)
    sz = [1 1];
elseif isscalar(args) && isscalar(args{1})
    sz = [args{1} args{1}];
elseif isscalar(args)
    sz = args{1}(:)';
else
    sz = [args{:}];
end
end
