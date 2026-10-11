function export_fixtures(matlab_repo)
% export_fixtures - write tests/fixtures/<namespace>.mat from the MATLAB toolkit: for
% each ported function, a cell array of cases, each holding the inputs and outputs in
% the fields in1, in2, ... and out1, out2, .... A function that draws random numbers
% runs on the replay stream (replay_start), and its cases also hold the seed and the
% number of uniforms used.
%
%   export_fixtures                 % the MATLAB toolkit in ../bvar-toolkit
%   export_fixtures(matlab_repo)
%
% Usage, from the root of this repository:
%   matlab -batch "addpath('tools/matlab'); export_fixtures"
% The toolkit's core/ must have no uncommitted changes; each file records its commit.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
if nargin < 1
    matlab_repo = fullfile(fileparts(root), 'bvar-toolkit');
end
[s1, sha] = system(sprintf('git -C "%s" rev-parse HEAD', matlab_repo));
[s2, dirty] = system(sprintf('git -C "%s" status --porcelain -- core', matlab_repo));
assert(s1 == 0 && s2 == 0, 'git failed in %s', matlab_repo);
assert(isempty(strtrim(dirty)), 'core/ of %s has uncommitted changes', matlab_repo);
run(fullfile(matlab_repo, 'setup.m'));
addpath(here);

meta = struct('matlab_commit', strtrim(sha), 'matlab_release', version('-release'), ...
    'exported', char(datetime('now', 'Format', 'yyyy-MM-dd')));
groups = {'util', @util_cases; 'diag', @diag_cases; 'priors', @priors_cases; ...
    'ml', @ml_cases; 'samplers', @samplers_cases; 'structural', @structural_cases; ...
    'forecast', @forecast_cases; 'optim', @optim_cases};
for g = 1:size(groups, 1)
    fx = groups{g, 2}();
    fx.meta = meta;
    save(fullfile(root, 'tests', 'fixtures', [groups{g, 1} '.mat']), '-struct', 'fx', '-v7');
    fprintf('%-10s %d functions\n', groups{g, 1}, numel(fieldnames(fx)) - 1);
end
end

%% ---- cases by namespace ----

function fx = util_cases()
rng(1, 'twister');
Yf = randn(12, 3);
fx.build_lags = {run_case(@bvar.util.build_lags, {Yf, 1}, 2), ...
    run_case(@bvar.util.build_lags, {Yf, 2}, 2), run_case(@bvar.util.build_lags, {Yf, 4}, 2)};
fx.diffmat = {run_case(@bvar.util.diffmat, {1}, 1), run_case(@bvar.util.diffmat, {6}, 1), ...
    run_case(@bvar.util.diffmat, {6, .7}, 1), run_case(@bvar.util.diffmat, {6, -.4}, 1)};
fx.dirirnd = {replay_case(11, @bvar.util.dirirnd, {[.5; 2; 3]}, 1), ...
    replay_case(12, @bvar.util.dirirnd, {[.5; 2; 3; 1.5], 4}, 1)};
Lv = exp(cumsum(.01*randn(10, 3) + .005));
Lv([4 5], 2) = NaN;  Lv([1 6 10], 3) = NaN;
fx.dlog_gaps = {run_case(@bvar.util.dlog_gaps, {Lv}, 3)};
G = double(rand(60, 4) > .6);
Gtie = [repmat([1 0 1 0], 3, 1); repmat([0 1 1 0], 3, 1); 1 1 1 1];
fx.gam_mode = {run_case(@bvar.util.gam_mode, {G}, 1), run_case(@bvar.util.gam_mode, {Gtie}, 1)};
fx.igrnd = {replay_case(13, @bvar.util.igrnd, {3, 2}, 1), ...
    replay_case(14, @bvar.util.igrnd, {[2; 3; 4], 1.5}, 1), ...
    replay_case(15, @bvar.util.igrnd, {[2 3 4; 5 6 7], [1 2 3; 4 5 6]}, 1)};
fx.kron_gram = {run_case(@bvar.util.kron_gram, {randn(15, 4), randn(3), exp(randn(15, 3)), randn(15, 3)}, 3)};
M5 = randn(5);  M30 = randn(30);
fx.ldet = {run_case(@bvar.util.ldet, {M5*M5' + 5*eye(5)}, 1), ...
    run_case(@bvar.util.ldet, {M30*M30' + eye(30)}, 1)};
x = 3*randn(6, 4);  x(:, 2) = -Inf;  x(3, 3) = Inf;
fx.logsumexp = {run_case(@bvar.util.logsumexp, {x}, 1), run_case(@bvar.util.logsumexp, {x, 2}, 1), ...
    run_case(@bvar.util.logsumexp, {randn(7, 1)}, 1)};
fx.mgammaln = {run_case(@bvar.util.mgammaln, {3, 5.5}, 1), ...
    run_case(@bvar.util.mgammaln, {10, 20.25}, 1), run_case(@bvar.util.mgammaln, {1, .7}, 1)};
Xm = randn(30, 3);  Xm(2, 1) = NaN;
Xm(:, 2) = NaN;  Xm(3:3:30, 2) = randn(10, 1);
Xm(:, 3) = NaN;  Xm(6:3:30, 3) = randn(9, 1);
fx.mm_constraint = {run_case(@bvar.util.mm_constraint, {Xm, [false true true]}, 4), ...
    run_case(@bvar.util.mm_constraint, {Xm, [false true false]}, 4, struct('weights', [1 1 1]/3))};
Ys = randn(5, 3);  Ys([2 7 11]) = NaN;
fx.select_obs = {run_case(@bvar.util.select_obs, {Ys}, 3), run_case(@bvar.util.select_obs, {randn(4, 2)}, 3)};
Xs = randn(5, 3);  Xs(2, 2) = 0;
fx.surform = {run_case(@bvar.util.surform, {Xs}, 1)};
fx.surform2 = {run_case(@bvar.util.surform2, {Xs, 2}, 1), run_case(@bvar.util.surform2, {randn(4, 2), 3}, 1)};
fx.tnormrnd = {replay_case(16, @bvar.util.tnormrnd, {randn(5, 1), rand(5, 1) + .5, -1, 2}, 1), ...
    replay_case(17, @bvar.util.tnormrnd, {.5, 2, 0, Inf, 6}, 1), ...
    replay_case(18, @bvar.util.tnormrnd, {zeros(4, 1), ones(4, 1), [-1; -2; 0; 1], [1; 0; 3; Inf]}, 1)};
fx.anormrnd = {};
for s = 19:26
    fx.anormrnd{end+1} = replay_case(s, @bvar.util.anormrnd, {randn, .2 + rand}, 1);
end
Yz = randn(4);  Yz(3, 1) = 0;
fx.vec = {run_case(@bvar.util.vec, {randn(4, 3)}, 1)};
fx.vech = {run_case(@bvar.util.vech, {randn(4)}, 1), run_case(@bvar.util.vech, {Yz}, 1)};
end

function fx = diag_cases()
rng(2, 'twister');
R = 600;  e = randn(R, 3);  x = zeros(R, 3);
for t = 2:R, x(t, :) = [.8 .3 -.2].*x(t-1, :) + e(t, :); end
fx.specvar0 = {run_case(@bvar.diag.specvar0, {x, 0}, 1), run_case(@bvar.diag.specvar0, {x, 10}, 1), ...
    run_case(@bvar.diag.specvar0, {x(:, 1)', 50}, 1)};
fx.inefficiency_factor = {run_case(@bvar.diag.inefficiency_factor, {x, 20}, 1)};
fx.mcse = {run_case(@bvar.diag.mcse, {x, 20}, 1)};
fx.geweke = {run_case(@bvar.diag.geweke, {x}, 3), run_case(@bvar.diag.geweke, {x, .2, .6, 5}, 3), ...
    run_case(@bvar.diag.geweke, {x(:, 2)}, 3)};
end

function fx = priors_cases()
[Y0, Y] = var_data(3, 4, 70, 8);
p = 2;  n = 4;
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
kap = [.3 .01 1 100];
fx.resid_var_ar4 = {run_case(@bvar.priors.resid_var_ar4, {Y0, Y}, 1)};
fx.acp_stru = {run_case(@bvar.priors.acp_stru, {n, p, kap, sig2}, 1), ...
    run_case(@bvar.priors.acp_stru, {n, p, kap, sig2, [1 3]}, 1)};
fx.acp_redu = {run_case(@bvar.priors.acp_redu, {n, p, kap, sig2}, 1), ...
    run_case(@bvar.priors.acp_redu, {n, p, kap, sig2, [1 3]}, 1)};
fx.acp_opt_kappa = {run_case(@bvar.priors.acp_opt_kappa, {Y0, Y, Z, p, [.04 .04], 'stru', []}, 2), ...
    run_case(@bvar.priors.acp_opt_kappa, {Y0, Y, Z, p, [], 'stru', []}, 2, struct('symmetric', true)), ...
    run_case(@bvar.priors.acp_opt_kappa, {Y0, Y, Z, p, [.04 .0016], 'redu', [1 3]}, 2), ...
    run_case(@bvar.priors.acp_opt_kappa, {Y0, Y, Z, p, [], 'redu', [1 3]}, 2, struct('symmetric', true)), ...
    run_case(@bvar.priors.acp_opt_kappa, {Y0, Y, Z, p, [.04 .04], 'stru', []}, 2, struct('ridge', 1e-6))};
end

function fx = ml_cases()
[Y0, Y] = var_data(4, 5, 80, 8);
p = 3;  n = 5;
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
pr1 = bvar.priors.acp_stru(n, p, [.2 .005 1 100], sig2);
pr2 = bvar.priors.acp_redu(n, p, [.2 .005 1 100], sig2, [2 4]);
fx.acp = {run_case(@bvar.ml.acp, {p, Y, Z, pr1}, 1), ...
    run_case(@bvar.ml.acp, {p, Y, Z, pr1}, 1, struct('ridge', 1e-6)), run_case(@bvar.ml.acp, {p, Y, Z, pr2}, 1)};
end

function fx = samplers_cases()
[Y0, Y] = var_data(5, 4, 60, 6);
p = 2;  n = 4;
pr = bvar.priors.acp_stru(n, p, [.3 .01 1 100], bvar.priors.resid_var_ar4(Y0, Y));
fx.acp_theta_sig = {replay_case(22, @bvar.samplers.acp_theta_sig, {Y0, Y, p, pr, 7}, 3), ...
    replay_case(23, @bvar.samplers.acp_theta_sig, {Y0, Y, p, pr, 2}, 3)};
end

function fx = structural_cases()
rng(6, 'twister');
n = 4;  p = 2;  nsim = 6;
fx.reduced_form = {run_case(@bvar.structural.reduced_form, ...
    {randn(nsim, n*(n-1)/2), randn(nsim, n^2*p + n), rand(nsim, n) + .5}, 2)};
end

function fx = forecast_cases()
rng(7, 'twister');
fx.predictive = {};
for nnp = [3 2; 2 1; 4 3]'
    n = nnp(1);  p = nnp(2);  nsim = 5;
    [A, Sig] = bvar.structural.reduced_form(.3*randn(nsim, n*(n-1)/2), .2*randn(nsim, n^2*p + n), ...
        rand(nsim, n) + .5);
    ylag = randn(n, p);
    fx.predictive{end+1} = run_case(@bvar.forecast.predictive, {A, Sig, ylag, 5}, 2);
    fx.predictive{end+1} = run_case(@bvar.forecast.predictive, {A, Sig, ylag, 3, [1 n]}, 2);
end
end

function fx = optim_cases()
% the objectives are written out in tests/test_optim.py in the same order of operations
f1 = @(x) (x - .3)^2*(x + .5) + .01/x;
f2 = @(x) x*x*x*x - 3*x*x*x + 2*x + 1;
fx.fminbnd = {opt_case('fminbnd', 'f1', f1, {0, 1}), opt_case('fminbnd', 'f2', f2, {-1, 3}), ...
    opt_case('fminbnd', 'f2', f2, {0, .5})};
rosen = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
quad3 = @(x) (x(1) - 1)^2 + 10*(x(2) - 2)^2 + 100*(x(3) + .5)^2 + x(1)*x(2);
fx.fminsearch = {opt_case('fminsearch', 'rosen', rosen, {[-1.2 1]}), ...
    opt_case('fminsearch', 'quad3', quad3, {[0 0 0]}), opt_case('fminsearch', 'quad3', quad3, {[1 2 -1]})};
end

%% ---- helpers ----

function c = run_case(fn, in, nout, opts)
% the outputs of fn(in{:}, name-value options), with the built-in random numbers
if nargin < 4, opts = struct(); end
nv = reshape([fieldnames(opts)'; struct2cell(opts)'], 1, []);
out = cell(1, nout);
[out{:}] = fn(in{:}, nv{:});
c = pack(in, out);
c.opts = opts;
end

function c = replay_case(seed, fn, in, nout)
% the outputs of fn(in{:}) on the replay stream of the seed
replay_start(seed);
cleanup = onCleanup(@() replay_stop());
out = cell(1, nout);
[out{:}] = fn(in{:});
c = pack(in, out);
c.opts = struct();
c.seed = seed;
c.count = replay_stop();
end

function c = opt_case(solver, name, f, in)
[x, fval] = feval(solver, f, in{:});
c = pack(in, {x, fval});
c.name = name;
end

function c = pack(in, out)
% one field per input and output, in1, in2, ... and out1, out2, ..., which loadmat
% reads back without the ambiguities of a cell array
c.nin = numel(in);
c.nout = numel(out);
for i = 1:numel(in), c.(sprintf('in%d', i)) = in{i}; end
for i = 1:numel(out), c.(sprintf('out%d', i)) = out{i}; end
end

function [Y0, Y] = var_data(seed, n, T, n0)
% generated data from a stable VAR(1) with an intercept
rng(seed, 'twister');
A1 = .5*eye(n) + .1*randn(n);
c = .1*randn(n, 1);
y = zeros(n0 + T, n);
for t = 2:n0 + T
    y(t, :) = (c + A1*y(t-1, :)' + .5*randn(n, 1))';
end
Y0 = y(1:n0, :);
Y = y(n0+1:end, :);
end
