function export_fixtures(matlab_repo)
% export_fixtures - write tests/fixtures/<namespace>.mat from the MATLAB toolkit: for
% each ported function, a cell array of cases, each holding the inputs and outputs in
% the fields in1, in2, ... and out1, out2, .... A function that draws random numbers
% runs on the replay stream (replay_start), and its cases also hold the seed and the
% number of uniforms used. drivers.mat holds run_ml of the Chan (2023) replication package,
% which the volatility-specification tutorial ports.
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
    'ml', @ml_cases; 'samplers', @samplers_cases; 'sv', @sv_cases; 'models', @models_cases; ...
    'structural', @structural_cases; 'forecast', @forecast_cases; 'optim', @optim_cases; ...
    'replay', @replay_cases; 'drivers', @() drivers_cases(matlab_repo, root)};
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
Asim = [.1 .2; .5 .1; .1 .4; -.2 .05; .03 -.1];  Ssim = [1 .3; .3 .5];   % a stable VAR(2)
fx.simulate_var = {replay_case(27, @bvar.util.simulate_var, {Asim, Ssim, 6}, 2), ...
    replay_case(28, @bvar.util.simulate_var, {Asim, Ssim, 5, [.3 -.1; 1 .2]}, 2), ...
    replay_case(29, @bvar.util.simulate_var, {[.2 0 .1; .6 .1 0; -.1 .5 .2; 0 .2 .3], ...
    [1 .2 0; .2 .8 .1; 0 .1 .6], 4}, 2)};
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
[Cm, i1, i2] = bvar.priors.minnesota_C(n, p, sig2);
fx.minnesota_C = {run_case(@bvar.priors.minnesota_C, {n, p, sig2}, 3), ...
    run_case(@bvar.priors.minnesota_C, {3, 1, [1; 2; .5]}, 3)};
fx.vtheta = {run_case(@bvar.priors.vtheta, {i1, i2, [.1 .05 .2 100], Cm.*exp(randn(size(Cm))), sig2}, 2), ...
    run_case(@bvar.priors.vtheta, {i1, i2, [.1 .1 NaN 100], Cm, sig2}, 2)};
fx.minn = {run_case(@bvar.priors.minn, {p, .04, .0016, 100, Y0, Y, 4}, 4), ...
    run_case(@bvar.priors.minn, {3, .1, .01, 10, Y0, Y, 4}, 4)};
fx.impact_B0 = {run_case(@bvar.priors.impact_B0, {Y0, Y, .04}, 2), run_case(@bvar.priors.impact_B0, {Y0, Y, 1}, 2)};
fx.niw = {run_case(@bvar.priors.niw, {4, [.04 100], Y0, Y, 'largebvar_nc'}, 5), ...
    run_case(@bvar.priors.niw, {2, [.04 100], Y0, Y, 'mlvarsv_ncp'}, 5), ...
    run_case(@bvar.priors.niw, {4, [.05 1.5 10 2 .8], Y0, Y, 'opthyper_ncp'}, 5), ...
    run_case(@bvar.priors.niw, {4, [.04 100], Y0, Y, 'kron_script'}, 5)};
    % the ridge decides the fit when a third series is nearly the sum of the other two
rng(23, 'twister');
Yc = [Y0(:, 1:2); Y(:, 1:2)];
Yc(:, 3) = Yc(:, 1) + Yc(:, 2) + 1e-3*randn(size(Yc, 1), 1);
fx.resid_var_allvars_ridge = {run_case(@bvar.priors.resid_var_allvars_ridge, {Y0, Y}, 1), ...
    run_case(@bvar.priors.resid_var_allvars_ridge, {Y0(:, 1:2), Y(:, 1:2)}, 1), ...
    run_case(@bvar.priors.resid_var_allvars_ridge, {Yc(1:8, :), Yc(9:end, :)}, 1)};
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

    % the densities, and the importance density of a log-volatility path
fx.lgampdf = {run_case(@bvar.ml.lgampdf, {[.5 1 2.5], 2, 3}, 1), ...
    run_case(@bvar.ml.lgampdf, {.7, [1 2 3], [.5 1 2]}, 1)};
Kp = [2 -.5 0; -.5 2 -.3; 0 -.3 1.5];
fx.lmvnpdf_pcn = {run_case(@bvar.ml.lmvnpdf_pcn, {[.1; -.2; .3], [0; .1; -.1], Kp}, 1), ...
    run_case(@bvar.ml.lmvnpdf_pcn, {[.1; -.2; .3], [0; .1; -.1], sparse(Kp)}, 1)};
fx.ltnormpdf = {run_case(@bvar.ml.ltnormpdf, {[.2 .5 .9], .6, .04, -1, 1}, 1), ...
    run_case(@bvar.ml.ltnormpdf, {.95, .98, .0025, -1, 1}, 1), ...
    run_case(@bvar.ml.ltnormpdf, {[-.95 -.5 .3], -.9, .09, -1, 1}, 1)};
rng(13, 'twister');
Hd = zeros(200, 60);  Hd(:, 1) = .3/sqrt(1 - .81)*randn(200, 1);
for t = 2:60, Hd(:, t) = .9*Hd(:, t-1) + .3*randn(200, 1); end
Hd = Hd + .5*sin((1:60)/9);
fx.isden_arss = {run_case(@bvar.ml.isden_arss, {Hd}, 5)};

    % the four marginal likelihoods of the volatility-specification tutorial, on generated
    % data and generated posterior draws, with 100 importance draws
[Y0, Y] = var_data(14, 3, 60, 8);
p = 2;  n = 3;  T = 60;  nd = 100;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
Hc = .8*Hd(1:nd, :);
hparaC = [.9 + .05*rand(nd, 1), .05 + .02*rand(nd, 1)];
kappaC = .04*exp(.3*randn(nd, 1));
HyC = struct();
[HyC.A0, HyC.VA, HyC.nu0, HyC.S0] = bvar.priors.niw(p, [.04 100], Y0, Y, 'mlvarsv_ncp');
HyC.c0 = [1, 1/.2^2];  HyC.nuh = 3;  HyC.Sh = .2;  HyC.phi0 = .98;  HyC.Vphi = .05^2;
fx.mlvarsv_csv = {replay_case(111, @bvar.ml.mlvarsv_csv, {X, Y, Y0, 100, HyC, Hc, hparaC, kappaC, false}, 3), ...
    replay_case(112, @bvar.ml.mlvarsv_csv, {X, Y, Y0, 60, HyC, Hc, hparaC, kappaC, true}, 3)};

Hs = zeros(nd, T, n);
for i = 1:n, Hs(:, :, i) = .7*Hd(1:nd, :) + .1*i + .05*randn(nd, T); end
betaS = [.2 -.1 .3] + .05*randn(nd, 3);
hparaS = [[.1 -.2 .3] + .1*randn(nd, 3), .9 + .05*rand(nd, 3), .05 + .02*rand(nd, 3)];
kappaS = [.04 .0016 .04].*exp(.3*randn(nd, 3));
HyS = struct();
[HyS.alp0, HyS.Valp] = bvar.priors.minn(p, .04, .0016, 100, Y0, Y, 4);
[HyS.beta0, HyS.Vbeta] = bvar.priors.impact_B0(Y0, Y, .04);
HyS.c0 = [1, 1/.2^2; 1, 1/.2^2; 1, 1];
HyS.nuh = 3*ones(n, 1);  HyS.Sh = .2*ones(n, 1);  HyS.mu0 = zeros(n, 1);  HyS.Vmu = 100*ones(n, 1);
HyS.phi0 = .98*ones(n, 1);  HyS.Vphi = .05^2*ones(n, 1);
inS = {X, Y, Y0, 100, HyS, 2, Hs, betaS, hparaS, kappaS};
fx.mlvarsv_arsv_redu = {replay_case(113, @bvar.ml.mlvarsv_arsv_redu, [inS, {false, false}], 3), ...
    replay_case(114, @bvar.ml.mlvarsv_arsv_redu, [inS, {false, true}], 3, struct('gram', 'blocks')), ...
    replay_case(115, @bvar.ml.mlvarsv_arsv_redu, [inS, {true, false}], 3)};

o_grid = [1; linspace(2, 20, 31)'];
oS = ones(nd, T);  hit = rand(nd, T) < .05;
oS(hit) = o_grid(1 + randi(31, nnz(hit), 1));
poS = .05 + .03*rand(nd, 1);
HyO = HyS;  HyO.p0a = 10/4;  HyO.p0b = (1 - 1/16)*40;
inO = {X, Y, Y0, 100, HyO, 2, Hs, betaS, hparaS, kappaS, oS, poS, o_grid};
fx.mlvarsv_arsvo_redu = {replay_case(116, @bvar.ml.mlvarsv_arsvo_redu, [inO, {false, false}], 3), ...
    replay_case(117, @bvar.ml.mlvarsv_arsvo_redu, [inO, {false, false}], 3, struct('bugcompat', true)), ...
    replay_case(118, @bvar.ml.mlvarsv_arsvo_redu, [inO, {false, true}], 3, struct('gram', 'blocks'))};

fx.mlvarsv_fsv = {};
for r = [1 2]
    Hf = zeros(nd, T, n + r);
    for i = 1:n + r, Hf(:, :, i) = .6*Hd(1:nd, :) - .5 + .05*randn(nd, T); end
    kl = n*r - r*(r+1)/2;
    lF = .5 + .1*randn(nd, kl);
    hparaF = [-.5 + .1*randn(nd, n + r), .9 + .05*rand(nd, n + r), .05 + .02*rand(nd, n + r)];
    kappaF = [.04 .0016].*exp(.3*randn(nd, 2));
    HyF = struct();
    [HyF.alp0, HyF.Valp] = bvar.priors.minn(p, .04, .0016, 100, Y0, Y, 4);
    HyF.c0 = [1, 1/.2^2; 1, 1/.2^2];
    HyF.nuh = 3*ones(n + r, 1);  HyF.Sh = .2*ones(n + r, 1);  HyF.mu0 = zeros(n + r, 1);
    HyF.Vmu = 100*ones(n + r, 1);  HyF.phi0 = .98*ones(n + r, 1);  HyF.Vphi = .05^2*ones(n + r, 1);
    HyF.l0 = 0;  HyF.Vl = 1;
    inF = {X, Y, Y0, 100, HyF, 2, Hf, hparaF, lF, kappaF};
    if r == 1
        fx.mlvarsv_fsv{end+1} = replay_case(119, @bvar.ml.mlvarsv_fsv, [inF, {false, false}], 3);
        inF{6} = 1;
        fx.mlvarsv_fsv{end+1} = replay_case(120, @bvar.ml.mlvarsv_fsv, [inF, {false, false}], 3);
    else
        fx.mlvarsv_fsv{end+1} = replay_case(121, @bvar.ml.mlvarsv_fsv, [inF, {false, true}], 3, ...
            struct('gram', 'blocks'));
        fx.mlvarsv_fsv{end+1} = replay_case(122, @bvar.ml.mlvarsv_fsv, [inF, {true, false}], 3);
    end
end
    % four variables and two factors, where the free loadings fill L column by column in an
    % order that differs from row by row
[Y0, Y] = var_data(16, 4, 60, 8);
n = 4;  r = 2;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
Hf = zeros(nd, T, n + r);
for i = 1:n + r, Hf(:, :, i) = .6*Hd(1:nd, :) - .5 + .05*randn(nd, T); end
lF = .5 + .1*randn(nd, n*r - r*(r+1)/2);
hparaF = [-.5 + .1*randn(nd, n + r), .9 + .05*rand(nd, n + r), .05 + .02*rand(nd, n + r)];
kappaF = [.04 .0016].*exp(.3*randn(nd, 2));
[HyF.alp0, HyF.Valp] = bvar.priors.minn(p, .04, .0016, 100, Y0, Y, 4);
HyF.nuh = 3*ones(n + r, 1);  HyF.Sh = .2*ones(n + r, 1);  HyF.mu0 = zeros(n + r, 1);
HyF.Vmu = 100*ones(n + r, 1);  HyF.phi0 = .98*ones(n + r, 1);  HyF.Vphi = .05^2*ones(n + r, 1);
fx.mlvarsv_fsv{end+1} = replay_case(131, @bvar.ml.mlvarsv_fsv, ...
    {X, Y, Y0, 100, HyF, 2, Hf, hparaF, lF, kappaF, false, false}, 3);
fx = kron_cases(fx);
fx = mltvpsv_cases(fx);
end

function fx = kron_cases(fx)
% the densities, likelihoods and marginal likelihoods of the Kronecker family of Chan (2020),
% on generated data and generated posterior draws
[Y0, Y] = var_data(17, 3, 40, 4);
p = 1;  n = 3;  T = 40;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
k = size(X, 2);
Aols = (X'*X)\(X'*Y);
U = Y - X*Aols;
Sols = U'*U/T;
fx.linvgammpdf = {run_case(@bvar.ml.linvgammpdf, {[.5 1 2.5], 3, 2}, 1), ...
    run_case(@bvar.ml.linvgammpdf, {.7, [1 2 3], [.5 1 2]}, 1)};
A0 = zeros(k, n);  VA0 = [100; .2*ones(k - 1, 1)];  nu0 = n + 3;  S0 = eye(n);
fx.lniwpdf = {run_case(@bvar.ml.lniwpdf, {Aols, Sols, A0, sparse(1:k, 1:k, 1./VA0), nu0, S0}, 1), ...
    run_case(@bvar.ml.lniwpdf, {Aols, Sols, Aols + .01, X'*X + eye(k), nu0 + T, Sols*T}, 1)};
Pd = bvar.util.dirirnd([2; 3; 5], 40);
fx.ldiripdf = {run_case(@bvar.ml.ldiripdf, {Pd, [2 3 5]}, 1), ...
    run_case(@bvar.ml.ldiripdf, {Pd(1, :), [.5 1 1.5]}, 1)};
fx.dirifit = {run_case(@bvar.ml.dirifit, {Pd}, 2), ...
    run_case(@bvar.ml.dirifit, {bvar.util.dirirnd([20; 1], 30)}, 2)};
hU = .3*sin((1:T)'/6);
fx.llike_ma = {run_case(@bvar.ml.llike_ma, {.3, U, Sols}, 1), run_case(@bvar.ml.llike_ma, {-.6, U, Sols}, 1)};
fx.llike_csv_ma = {run_case(@bvar.ml.llike_csv_ma, {.3, U, Sols, hU}, 1), ...
    run_case(@bvar.ml.llike_csv_ma, {-.6, U, Sols, hU - .2}, 1)};
fx.intlike_csv = {replay_case(221, @bvar.ml.intlike_csv, {Y, X, Aols, Sols, .9, .1, 30}, 2)};
fx.intlike_csv_ma = {replay_case(222, @bvar.ml.intlike_csv_ma, {Y, X, Aols, Sols, .3, .9, .1, 30}, 2)};
fx.intlike_t_csv = {replay_case(223, @bvar.ml.intlike_t_csv, {Y, X, Aols, Sols, .9, .1, 8, 30}, 2)};
fx.intlike_csv_t_ma = {replay_case(224, @bvar.ml.intlike_csv_t_ma, {Y, X, Aols, Sols, .3, .9, .1, 8, 30}, 2)};

    % the closed form, and generated stores of eight draws for the others; the stores of the
    % t-MA model and the reduced runs of the MA models are long enough for the second
    % search of the psi proposal, at sweep 100
pri = struct('A0', A0, 'VA0', VA0, 'nu0', nu0, 'S0', S0);
KA = sparse(1:k, 1:k, 1./VA0) + X'*X;
Ahat = KA\(A0./VA0 + X'*Y);
Shat = S0 + A0'*(A0./VA0) + Y'*Y - Ahat'*KA*Ahat;
fx.kron_bvar = {run_case(@bvar.ml.kron_bvar, {Y, X, pri, struct('Ahat', Ahat, 'Shat', (Shat + Shat')/2, 'KA', KA)}, 2)};
G = kron_store(8, Aols, Sols, T);
priT = pri;  priT.nuub = 50;
fx.kron_bvar_t = {run_case(@bvar.ml.kron_bvar_t, {Y, X, priT, struct('nsims', 8, 'store_A', G.store_A, ...
    'store_Sig', G.store_Sig, 'store_nu', G.nu, 'store_lam', G.store_lam)}, 2)};
priM = pri;  priM.psi0 = 0;  priM.Vpsi = 1;
estM = struct('nsims', 8, 'store_A', G.store_A, 'store_Sig', G.store_Sig, 'store_psi', G.psi, ...
    'state', struct('psi', G.psi(end) + .01));
fx.kron_bvar_ma = {run_case(@bvar.ml.kron_bvar_ma, {Y, X, priM, estM}, 2), ...
    run_case(@bvar.ml.kron_bvar_ma, {Y, X, priM, estM}, 2, struct('bugcompat', true))};
priC = pri;  priC.rho0 = .9;  priC.Vrho = .2^2;  priC.nuh0 = 5;  priC.Sh0 = .4;
estC = struct('nsims', 8, 'store_A', G.store_A, 'store_Sig', G.store_Sig, 'store_h', G.store_h, ...
    'store_theta', [G.rho G.sigh2], 'state', struct('countrho', 7));
fx.kron_bvar_csv = {replay_case(225, @bvar.ml.kron_bvar_csv, {Y, X, priC, estC}, 2, struct('R', 30))};
priTC = priC;  priTC.nuub = 50;
estTC = struct('nsims', 8, 'store_A', G.store_A, 'store_Sig', G.store_Sig, 'store_h', G.store_h, ...
    'store_lam', G.store_lam, 'store_theta', [G.nu G.rho G.sigh2]);
fx.kron_bvar_t_csv = {replay_case(226, @bvar.ml.kron_bvar_t_csv, {Y, X, priTC, estTC}, 2, struct('R', 30))};
G101 = kron_store(101, Aols, Sols, T);
priTM = priM;  priTM.nuub = 50;
estTM = struct('nsims', 101, 'store_A', G101.store_A, 'store_Sig', G101.store_Sig, ...
    'store_lam', G101.store_lam, 'store_theta', [G101.psi G101.nu], 'state', struct('psihat', .25));
fx.kron_bvar_t_ma = {replay_case(227, @bvar.ml.kron_bvar_t_ma, {Y, X, priTM, estTM}, 2)};
priCM = priC;  priCM.psi0 = 0;  priCM.Vpsi = 1;
estCM = struct('nsims', 8, 'store_A', G.store_A, 'store_Sig', G.store_Sig, 'store_h', G.store_h, ...
    'store_theta', [G.psi G.rho G.sigh2], 'state', struct('psihat', .25));
fx.kron_bvar_csv_ma = {replay_case(228, @bvar.ml.kron_bvar_csv_ma, {Y, X, priCM, estCM}, 2, ...
    struct('R', 30, 'nsims2', 101))};
priCTM = priCM;  priCTM.nuub = 50;
W = .05*randn(n);
estCTM = struct('nsims', 8, 'store_A', G.store_A, 'store_Sig', G.store_Sig, 'store_h', G.store_h, ...
    'store_lam', G.store_lam, 'store_theta', [G.psi G.rho G.sigh2 G.nu], ...
    'state', struct('psihat', .25, 'psi', G.psi(end) - .02, 'Sig', Sols + W*W'));
fx.kron_bvar_csv_t_ma = {replay_case(229, @bvar.ml.kron_bvar_csv_t_ma, {Y, X, priCTM, estCTM}, 2, ...
        struct('R', 30, 'nsims2', 101)), ...
    replay_case(230, @bvar.ml.kron_bvar_csv_t_ma, {Y, X, priCTM, estCTM}, 2, ...
        struct('R', 30, 'nsims2', 12, 'bugcompat', true))};
end

function G = kron_store(nsims, A, Sig, T)
% generated stores of nsims posterior draws near (A, Sig): the running sums of A and Sig, and
% the paths and parameters of the volatility, mixture-scale and MA blocks
G.store_A = zeros(size(A));  G.store_Sig = zeros(size(Sig));
for i = 1:nsims
    G.store_A = G.store_A + A + .02*randn(size(A));
    W = .05*randn(size(Sig));
    G.store_Sig = G.store_Sig + Sig + W*W';
end
G.store_h = .3*sin((1:T)/6) + .1*randn(nsims, T);
G.store_lam = 1./gamrnd(4, 1/4, nsims, T);
G.psi = .3 + .05*randn(nsims, 1);
G.rho = .9 + .03*randn(nsims, 1);
G.sigh2 = .1 + .02*rand(nsims, 1);
G.nu = 8 + rand(nsims, 1);
end

function fx = mltvpsv_cases(fx)
% the importance sampling estimators of Chan and Eisenstat (2018) and the integrated
% likelihoods they use, for two variables and one lag on generated data, with generated
% posterior draws and 40 importance draws. The log priors are written out in
% tests/test_mltvpsv.py in the same order of operations.
[Y0, Yd] = var_data(18, 2, 20, 2);
n = 2;  T = 20;  nd = 50;  M = 40;
xt = [ones(T, 1), [Y0(end, :); Yd(1:end-1, :)]];
k = n*size(xt, 2);
Y = reshape(Yd', [], 1);                    % stacked by period
bigX = zeros(T*n, k);  bigXt = sparse(T*n, T*k);
for t = 1:T
    bigX((t-1)*n + (1:n), :) = kron(eye(n), xt(t, :));
    bigXt((t-1)*n + (1:n), (t-1)*k + (1:k)) = kron(eye(n), xt(t, :));
end
theta = bigX\Y;
s = var(reshape(Y - bigX*theta, n, T)')';
bigX = sparse(bigX);
fx.intlike_tvp = {run_case(@bvar.ml.intlike_tvp, {Y, s, .01*ones(k, 1), bigXt, theta}, 1)};
    % the importance draws of the integrated likelihoods: ten, more when their log weights
    % vary much (the last case of each), and a given number
fx.intlike_cvarsv = {replay_case(242, @bvar.ml.intlike_cvarsv, {Y, theta, [.05; .05], bigX, log(s)}, 1), ...
    replay_case(243, @bvar.ml.intlike_cvarsv, {Y, theta, [.05; .05], bigX, log(s), 25}, 1), ...
    replay_case(300, @bvar.ml.intlike_cvarsv, {Y, theta, [5; 5], bigX, log(s)}, 1)};
fx.intlike_tvpsv = {replay_case(244, @bvar.ml.intlike_tvpsv, {Y, .01*ones(k, 1), [.05; .05], bigXt, log(s), theta}, 2), ...
    replay_case(245, @bvar.ml.intlike_tvpsv, {Y, .01*ones(k, 1), [.05; .05], bigXt, log(s), theta, 15}, 2), ...
    replay_case(310, @bvar.ml.intlike_tvpsv, {Y, ones(k, 1), [30; 30], bigXt, log(s), theta}, 2)};
fx.intlike_rsvar = {run_case(@bvar.ml.intlike_rsvar, {Yd, bigX, [theta, theta + .1], [s, 2*s], [.8 .2; .3 .7]}, 1), ...
    run_case(@bvar.ml.intlike_rsvar, {Yd, bigX, [theta, theta + .1], [s, 2*s], [.8 .2; .3 .7], 1/3}, 1)};

lpt = @(th) -.5*sum(th.^2)/10;              % log prior kernels of coefficients,
lps = @(v) -sum(2*log(v) + .1./v);          % variances
lpP = @(P) log(P(1, 1)) + log(P(2, 2));     % and transition matrices
store_theta = theta' + .02*randn(nd, k);
store_Sig = s'.*exp(.1*randn(nd, n));
store_Sigh = .05*exp(.2*randn(nd, n));
store_h0 = log(s') + .1*randn(nd, n);
store_Sigtheta = .01*exp(.2*randn(nd, k));
store_theta0 = theta' + .05*randn(nd, k);
fx.mltvpsv_cvar = {replay_case(231, @(Y, a, b, bX, M) bvar.ml.mltvpsv_cvar(Y, a, b, ...
    @(th, sg) lpt(th) + lps(sg), bX, M), {Y, store_theta, store_Sig, bigX, M}, 3)};
fx.mltvpsv_cvarsv = {replay_case(232, @(Y, a, b, c, bX, M) bvar.ml.mltvpsv_cvarsv(Y, a, b, c, ...
    @(th, sh, h0) lpt(th) + lps(sh) + lpt(h0), bX, M), {Y, store_theta, store_Sigh, store_h0, bigX, M}, 3)};
fx.mltvpsv_tvp = {replay_case(233, @(Y, a, b, c, bX, M) bvar.ml.mltvpsv_tvp(Y, a, b, c, ...
    @(sg, st, t0) lps(sg) + lps(st) + lpt(t0), bX, M), {Y, store_Sig, store_Sigtheta, store_theta0, bigXt, M}, 3)};
fx.mltvpsv_tvpsv = {replay_case(234, @(Y, a, b, c, d, bX, M) bvar.ml.mltvpsv_tvpsv(Y, a, b, c, d, ...
    @(st, sh, t0, h0) lps(st) + lps(sh) + lpt(t0) + lpt(h0), bX, M), ...
    {Y, store_Sigtheta, store_Sigh, store_h0, store_theta0, bigXt, M}, 3)};

    % two regimes, with coefficients and variances that switch, and with one of them shared
store_thetaR = [store_theta, theta' + .1 + .02*randn(nd, k)];
store_SigR = [store_Sig, 2*store_Sig.*exp(.1*randn(nd, n))];
store_P = zeros(nd, 2, 2);
store_P(:, 1, :) = bvar.util.dirirnd([8; 2], nd);
store_P(:, 2, :) = bvar.util.dirirnd([3; 7], nd);
rs = @(fn) @(Y, a, b, c, bX, M, bc) fn(Y, a, b, c, @(th, sg, P) lpt(th) + lps(sg) + lpP(P), bX, M, bc);
fx.mltvpsv_rs = {replay_case(235, rs(@bvar.ml.mltvpsv_rs), {Y, store_thetaR, store_SigR, store_P, bigX, M, false}, 3), ...
    replay_case(236, rs(@bvar.ml.mltvpsv_rs), {Y, store_thetaR, store_SigR, store_P, bigX, M, true}, 3)};
fx.mltvpsv_rs_r1 = {replay_case(237, rs(@bvar.ml.mltvpsv_rs_r1), {Y, store_theta, store_SigR, store_P, bigX, M, false}, 3)};
fx.mltvpsv_rs_r2 = {replay_case(238, rs(@bvar.ml.mltvpsv_rs_r2), {Y, store_thetaR, store_Sig, store_P, bigX, M, false}, 3)};

    % the impact coefficient of the second equation on the first variable: time-varying with
    % constant VAR coefficients (R1), constant with time-varying ones (R2), and constant with
    % time-varying intercepts (R3)
W = sparse(T*n, T);
for t = 1:T, W((t-1)*n + 2, t) = -Yd(t, 1); end
W2 = full(sum(W, 2));
store_Sigg = .01*exp(.2*randn(nd, 1));
store_g = .2 + .05*randn(nd, 1);
store_Sigmu = .01*exp(.2*randn(nd, n));
store_mu0 = .1*randn(nd, n);
fx.mltvpsv_tvp_r1_sv = {replay_case(239, @(Y, a, b, c, d, e, Xt, W, M) bvar.ml.mltvpsv_tvp_r1_sv(Y, a, b, c, d, e, ...
    @(b, sg, sh, g0, h0) lpt(b) + lps(sg) + lps(sh) + lpt(g0) + lpt(h0), Xt, W, M), ...
    {Y, store_theta, store_Sigg, store_Sigh, store_h0, store_g, bigX, W, M}, 3)};
fx.mltvpsv_tvp_r2_sv = {replay_case(240, @(Y, a, b, c, d, e, Xt, W, M) bvar.ml.mltvpsv_tvp_r2_sv(Y, a, b, c, d, e, ...
    @(g, sb, sh, b0, h0) lpt(g) + lps(sb) + lps(sh) + lpt(b0) + lpt(h0), Xt, W, M), ...
    {Y, store_g, store_Sigtheta, store_Sigh, store_h0, store_theta0, bigXt, W2, M}, 3)};
fx.mltvpsv_tvp_r3_sv = {replay_case(241, @(Y, a, b, c, d, e, f, Z, M) bvar.ml.mltvpsv_tvp_r3_sv(Y, a, b, c, d, e, f, ...
    @(b, g, sm, sh, m0, h0) lpt(b) + lpt(g) + lps(sm) + lps(sh) + lpt(m0) + lpt(h0), Z, M), ...
    {Y, store_theta, store_g, store_Sigmu, store_Sigh, store_h0, store_mu0, [bigX, sparse(W2)], M}, 3)};
end

function fx = samplers_cases()
[Y0, Y] = var_data(5, 4, 60, 6);
p = 2;  n = 4;
pr = bvar.priors.acp_stru(n, p, [.3 .01 1 100], bvar.priors.resid_var_ar4(Y0, Y));
fx.acp_theta_sig = {replay_case(22, @bvar.samplers.acp_theta_sig, {Y0, Y, p, pr, 7}, 3), ...
    replay_case(23, @bvar.samplers.acp_theta_sig, {Y0, Y, p, pr, 2}, 3)};

    % var_coef_joint with a sparse diagonal and a dense prior precision
[Yv, Xv] = bvar.util.build_lags([Y0(end-p+1:end, :); Y], p);
kn = size(Xv, 2)*n;
w = exp(-.3*sin((1:size(Yv, 1))'/5));
Sv = cov(Yv) + .1*eye(n);
P = randn(kn);
fx.var_coef_joint = {replay_case(47, @bvar.samplers.var_coef_joint, ...
    {Yv, Xv, Sv, w, sparse(1:kn, 1:kn, 1./(.5 + rand(kn, 1)))}, 2), ...
    replay_case(48, @bvar.samplers.var_coef_joint, {Yv, Xv, Sv, ones(size(Yv, 1), 1), P*P'/kn + eye(kn)}, 2)};

    % missing_var without restrictions, with hard ones (one draw, and three draws with no
    % volatility and their own initial-condition priors), with soft ones (one variance, and
    % one per row) and with no missing value
[Ym, M, z, ~, A, Sig, h, Yfull] = mf_problem(41, 40);
q = size(M, 1);
fx.missing_var = {replay_case(41, @bvar.samplers.missing_var, {Ym, A, Sig, h, 2}, 3), ...
    replay_case(42, @bvar.samplers.missing_var, {Ym, A, Sig, h, 2}, 3, struct('M', M, 'z', z)), ...
    replay_case(43, @bvar.samplers.missing_var, {Ym, A, Sig, [], 2}, 3, ...
        struct('M', M, 'z', z, 'ndraws', 3, 'V0', [50 100 10], 'm0', [.1 0 -.2])), ...
    replay_case(44, @bvar.samplers.missing_var, {Ym, A, Sig, h, 2}, 3, struct('M', M, 'z', z, 'O', .01)), ...
    replay_case(45, @bvar.samplers.missing_var, {Ym, A, Sig, h, 2}, 3, ...
        struct('M', M, 'z', z, 'O', .001*(1:q)')), ...
    replay_case(46, @bvar.samplers.missing_var, {Yfull, A, Sig, h, 2}, 3)};

    % the blocks of bvar.models.var_sv, at volatilities that move over time
T = size(Y, 1);  k = 1 + n*p;
hv = .4*sin((1:T)'/6 + (1:n)) - .2;
B0v = eye(n) + .2*triu(randn(n)) - .1*tril(randn(n), -1);
Ainit = .1*randn(k, n);
[Cm, i1, i2] = bvar.priors.minnesota_C(n, p, bvar.priors.resid_var_ar4(Y0, Y));
[~, dV] = bvar.priors.vtheta(i1, i2, [.1 .05 NaN 100], Cm, ones(n, 1));
fx.eq_var_oi = {replay_case(81, @bvar.samplers.eq_var_oi, {Y, Xv, B0v, hv, Ainit, dV}, 1)};
Atri = eye(n);  Atri(logical(tril(ones(n), -1))) = .3*randn(n*(n-1)/2, 1);
Bv = Ainit';
fx.eq_tri_cs = {replay_case(82, @bvar.samplers.eq_tri_cs, {Y, Xv, Xv*Bv', Bv, Atri, hv, dV, zeros(k*n, 1)}, 2), ...
    replay_case(83, @bvar.samplers.eq_tri_cs, {Y, Xv, Xv*Bv', Bv, Atri, hv, dV, .05*randn(k*n, 1)}, 2)};
E = Y - Xv*Ainit;
fx.alp_tri_cs = {replay_case(84, @bvar.samplers.alp_tri_cs, {E, hv, .5 + rand(n*(n-1)/2, 1)}, 1), ...
    replay_case(85, @bvar.samplers.alp_tri_cs, {E, hv, ones(n*(n-1)/2, 1), 1 + rand(T, 1)}, 1)};
fx.horseshoe_kappa_psi = {replay_case(86, @bvar.samplers.horseshoe_kappa_psi, ...
    {Ainit(:), i1, i2, Cm, [.1 .05 NaN 100], .5 + rand(numel(i1), 1), .5 + rand(numel(i2), 1), [.8; 1.2]}, 6)};

    % the blocks of the volatility-specification tutorial: the reduced-form coefficients
    % under a unit lower triangular impact matrix, with and without outlier scales, and the
    % factor path and the loadings of the factor model with one and two factors
[al0, Va] = bvar.priors.minn(p, .04, .0016, 100, Y0, Y, 4);
fx.eq_var_redu_tri = {replay_case(123, @bvar.samplers.eq_var_redu_tri, {Y, Xv, Atri, hv, Ainit, Va, al0}, 1), ...
    replay_case(124, @bvar.samplers.eq_var_redu_tri, {Y, Xv, Atri, hv, Ainit, Va, .05*randn(k*n, 1), ...
    1 + 3*(rand(T, 1) < .1)}, 1)};
fx.factor_fsv = {};  fx.eq_fsv_load = {};
for r = [1 2]
    Lf = [eye(r); .5 + .2*randn(n - r, r)];
    Lf(1:r, 1:r) = eye(r) + tril(.3*randn(r), -1);
    hf = [hv, .5*sin((1:T)'/4 + (1:r))];
    Ff = randn(T, r);
    fx.factor_fsv{end+1} = replay_case(124 + r, @bvar.samplers.factor_fsv, {Y, Xv, Ainit, Lf, hf}, 1);
    fx.eq_fsv_load{end+1} = replay_case(126 + r, @bvar.samplers.eq_fsv_load, ...
        {Y, Xv, Ff, hf, Ainit, Lf, Va, .02*randn(k*n, 1), 1, .1}, 2);
end

    % the structural-form coefficients of Chan (2021) equation by equation, and the
    % order-invariant draw of A
rng(20, 'twister');
fx.eq_gauss = {replay_case(201, @bvar.samplers.eq_gauss, {Y, Xv, hv, .5 + rand(n*(n-1)/2, 1), dV}, 3)};
fx.eq_svar_oi = {replay_case(202, @bvar.samplers.eq_svar_oi, {Y, Xv, B0v, hv, Ainit, dV}, 1)};

    % the hierarchical shrinkage block in its three variants and the normal-gamma shape, on
    % the replay stream, where gigrnd returns the mode of its distribution; a zero
    % coefficient sends its local scale to the floor
bg = Ainit(:);  bg(i2(1)) = 0;
psi1 = .5 + rand(numel(i1), 1);  psi2 = .5 + rand(numel(i2), 1);
fx.gig_shrinkage = {replay_case(203, @bvar.samplers.gig_shrinkage, ...
        {'mng', bg, i1, i2, Cm, [.1 .05 NaN 100], psi1, psi2, .5, [1 1], [1 2], n, p, 1e-10}, 3), ...
    replay_case(204, @bvar.samplers.gig_shrinkage, ...
        {'ng', bg, i1, i2, [], .1, psi1, psi2, .8, [1 1], [], n, p, 1e-16}, 3), ...
    replay_case(205, @bvar.samplers.gig_shrinkage, ...
        {'minn', bg, i1, i2, Cm, [.1 .05 NaN 100], psi1, psi2, [], [1 1], [1 2], n, p, []}, 3)};
fx.nu_psi_ng = {replay_case(206, @nu_psi_ng_grid, {psi1, psi2, 1, 1, [.5 1 2 4]}, 3), ...
    replay_case(207, @nu_psi_ng_grid, {.02 + .05*rand(numel(i1), 1), .02 + .05*rand(numel(i2), 1), ...
        .3, 1, [.1 .5 1]}, 3)};

    % the hybrid TVP block for the first equation, which has no impact coefficients, and for
    % the third, with the indicators drawn and at each of their four values
Xi3 = [Xv, -Y(:, 1:2)];
th1 = .1*randn(k, 1);  th3 = .1*randn(k + 2, 1);
S1 = .01*(.5 + rand(k, 1));  S3 = .01*(.5 + rand(k + 2, 1));
fx.eq_hyb_tvp = {replay_case(208, @bvar.samplers.eq_hyb_tvp, ...
        {Xv, th1, S1, Y(:, 1), hv(:, 1), [.4; .7], k, false, [0; 0]}, 5), ...
    replay_case(209, @bvar.samplers.eq_hyb_tvp, {Xi3, th3, S3, Y(:, 3), hv(:, 3), [.3; .6], k, false, [0; 0]}, 5)};
gams = [1 1; 1 0; 0 1; 0 0]';
for g = 1:4
    fx.eq_hyb_tvp{end+1} = replay_case(209 + g, @bvar.samplers.eq_hyb_tvp, ...
        {Xi3, th3, S3, Y(:, 3), hv(:, 3), [.3; .6], k, true, gams(:, g)}, 5);
end
end

function [nu_psi, flag, fgrid] = nu_psi_ng_grid(psi_kappa1, psi_kappa2, nu_psi, lam0, grid)
% bvar.samplers.nu_psi_ng, with its log target kernel evaluated on a grid in place of the handle
[nu_psi, flag, f] = bvar.samplers.nu_psi_ng(psi_kappa1, psi_kappa2, nu_psi, lam0);
fgrid = arrayfun(f, grid);
end

function [nu, flag, fgrid] = nu_studentt_grid(lam, nu, nu_ub, grid)
% bvar.sv.nu_studentt, with its log target kernel evaluated on a grid in place of the handle
[nu, flag, f] = bvar.sv.nu_studentt(lam, nu, nu_ub);
fgrid = arrayfun(f, grid);
end

function fx = sv_cases()
rng(8, 'twister');
T = 80;  n = 4;
h = zeros(T, 1);
for t = 2:T, h(t) = .95*h(t-1) + .2*randn; end
s2 = exp(h).*sum(randn(T, n).^2, 2);
h0 = h + .1*randn(T, 1);
    % blocks of 20 after a random first one, one block, forced acceptance, blocks of one,
    % and two with c_reject small enough for the Metropolis-Hastings step to reject
fx.csv_armh_block = {replay_case(51, @bvar.sv.csv_armh_block, {s2, .95, .04, zeros(T, 1), n}, 3, ...
        struct('block', 20)), ...
    replay_case(52, @bvar.sv.csv_armh_block, {s2, .95, .04, h0, n}, 3, struct('block', 100)), ...
    replay_case(53, @bvar.sv.csv_armh_block, {s2, .9, .1, h0, n}, 3, ...
        struct('block', 7, 'ForcedAccept', true, 'c_reject', 2)), ...
    replay_case(54, @bvar.sv.csv_armh_block, {s2(1:30), .95, .04, h0(1:30), n}, 3, struct('block', 1)), ...
    replay_case(65, @bvar.sv.csv_armh_block, {s2, .95, .04, h0, n}, 3, struct('block', 20, 'c_reject', 1)), ...
    replay_case(64, @bvar.sv.csv_armh_block, {s2, .95, .04, h0, n}, 3, struct('block', 100, 'c_reject', .1))};

Hyper = struct('nuh', 10, 'Sh', .004, 'phi0', .98, 'Vphi', .05^2);
H3 = zeros(T, 3);
for t = 2:T, H3(t, :) = [.9 .5 .97].*H3(t-1, :) + .2*randn(1, 3); end
    % phi_hat above the bound, below minus the bound and inside it, so that the truncated
    % candidate is drawn below the mean, above it and around it
Hb = zeros(400, 3);  Hb(1, :) = [1 1 .3];
for t = 2:400, Hb(t, :) = [1.05 -1.05 .5].*Hb(t-1, :) + .1*randn(1, 3); end
    % phi_hat about 44 standard deviations beyond each bound, where the normal cdf
    % underflows and the candidate comes from the tail sampler
He = ones(2000, 2);
for t = 2:2000, He(t, :) = [1.02 -1.02].*He(t-1, :) + .1*randn(1, 2); end
fx.sv0_params = {replay_case(55, @bvar.sv.sv0_params, {H3, [.9; .5; .95], Hyper}, 3), ...
    replay_case(56, @bvar.sv.sv0_params, {H3, [.9; .5; .95], Hyper, .99}, 3, struct('proposal', 'truncated')), ...
    replay_case(57, @bvar.sv.sv0_params, {h, .95, Hyper, 1}, 3, struct('proposal', 'truncated')), ...
    replay_case(58, @bvar.sv.sv0_params, {Hb, [.9; -.9; .5], Hyper}, 3), ...
    replay_case(59, @bvar.sv.sv0_params, {Hb, [.9899; -.9899; .5], Hyper, .99}, 3, struct('proposal', 'truncated')), ...
    replay_case(60, @bvar.sv.sv0_params, {He, [.99; -.99], Hyper, .99}, 3, struct('proposal', 'truncated'))};

    % the auxiliary mixture sampler, with mean zero and with a mean
ystar = log((exp(h/2).*randn(T, 1)).^2 + .0001);
fx.ksc_ar1_mean = {replay_case(87, @bvar.sv.ksc_ar1_mean, {ystar, h, 0, .95, .04}, 2), ...
    replay_case(88, @bvar.sv.ksc_ar1_mean, {ystar + 1, h0 + 1, 1, .9, .1}, 2)};
    % paths with means; a zero mean skips the mu draw; a last column with mean zero
Hy = struct('nuh', 3*ones(3, 1), 'Sh', .1*ones(3, 1), 'phi0', .95*ones(3, 1), ...
    'Vphi', .05^2*ones(3, 1), 'mu0', zeros(3, 1), 'Vmu', 100*ones(3, 1));
Hy2 = Hy;  Hy2.mu0 = zeros(2, 1);  Hy2.Vmu = 100*ones(2, 1);
Hm = H3 + [1 -1 .5];
fx.sv_params = {replay_case(89, @bvar.sv.sv_params, {Hm, [.8; -.8; .4], [.9; .5; .95], Hy}, 4), ...
    replay_case(90, @bvar.sv.sv_params, {Hm, [.8; 0; .4], [.9; .5; .95], Hy}, 4), ...
    replay_case(91, @bvar.sv.sv_params, {[Hm(:, 1:2) H3(:, 3)], [.8; -.8], [.9; .5; .95], Hy2, .998}, 4)};

    % the whole path in one block: the default, forced acceptance from another starting
    % point of the mode search, a small c_reject, and on the stream of seed 97 a small
    % c_reject for which the Metropolis step rejects, without and with forced acceptance
fx.csv_armh = {replay_case(97, @bvar.sv.csv_armh, {s2, .95, .04, h0, n}, 2), ...
    replay_case(98, @bvar.sv.csv_armh, {s2, .95, .04, h0, n, true, zeros(T, 1)}, 2), ...
    replay_case(99, @bvar.sv.csv_armh, {s2, .95, .04, h0, n}, 2, struct('c_reject', .1)), ...
    replay_case(97, @bvar.sv.csv_armh, {s2, .95, .04, h0, n}, 2, struct('c_reject', .1)), ...
    replay_case(97, @bvar.sv.csv_armh, {s2, .95, .04, h0, n, true}, 2, struct('c_reject', .1))};

    % starting values of a path, and the outlier block on generated residuals
fx.init_approx1N = {run_case(@bvar.sv.init_approx1N, {s2/n, 0, .95, .04}, 1), ...
    run_case(@bvar.sv.init_approx1N, {s2/n + .5, -1, .8, .2}, 1)};
[Y0o, Yo] = var_data(15, 3, 50, 6);
[~, Xo] = bvar.util.build_lags([Y0o(end-1:end, :); Yo], 2);
Ao = (Xo'*Xo + .1*eye(7))\(Xo'*Yo);
Yo(7, :) = Yo(7, :) + [4 -3 5];                       % an outlier
o_grid = [1; linspace(2, 20, 31)'];
B0o = [1 0 0; .3 1 0; -.2 .1 1];
ho = .2*sin((1:50)'/5 + (1:3));
fx.svo_outlier = {replay_case(129, @bvar.sv.svo_outlier, {Yo, Xo, Ao, B0o, ho, o_grid, 1/16, 10/4, (1-1/16)*40}, 2), ...
    replay_case(130, @bvar.sv.svo_outlier, {Yo, Xo, Ao, B0o, ho - 1, o_grid, .2, 1, 1}, 2)};

    % the auxiliary mixture sampler for a random walk, from a known and from a diffuse
    % initial value, and the Student-t degrees of freedom
rng(21, 'twister');
hr = cumsum(.1*randn(T, 1));
ystar_r = log((exp(hr/2).*randn(T, 1)).^2 + .0001);
fx.ksc_rw_h0 = {replay_case(214, @bvar.sv.ksc_rw_h0, {ystar_r, hr, .01, 0}, 1), ...
    replay_case(215, @bvar.sv.ksc_rw_h0, {ystar_r + 1, hr + 1, .05, .5}, 1)};
fx.ksc_rw_diffuse = {replay_case(216, @bvar.sv.ksc_rw_diffuse, {ystar_r, hr, .01, 10}, 2), ...
    replay_case(217, @bvar.sv.ksc_rw_diffuse, {ystar_r + 1, hr + 1, .05, 100}, 2)};
lam5 = 1./gamrnd(5/2, 2/5, T, 1);
lam30 = 1./gamrnd(15, 1/15, T, 1);
fx.nu_studentt = {replay_case(218, @nu_studentt_grid, {lam5, 6, 50, [3 5 10 20]}, 3), ...
    replay_case(219, @nu_studentt_grid, {lam30, 20, 30, [3 10 25]}, 3), ...
    replay_case(220, @nu_studentt_grid, {lam30, 20, 21, [3 10 20]}, 3)};
end

function fx = models_cases()
    % short chains: past the 20 sweeps of forced acceptance, with the draws returned, and
    % with h in one block and other settings
[Y, M, z, sig2] = mf_problem(61, 72);
o1 = struct('M', M, 'z', z, 'sig2', sig2, 'nsim', 8, 'burnin', 15, 'hblock', 15, 'draws', true);
o2 = struct('M', M, 'z', z, 'sig2', sig2, 'nsim', 3, 'burnin', 22, 'hblock', 100, ...
    'c_reject', 2, 'kappa', [.1 .02], 'V0', 50, 'm0', .1, 'quantiles', [.1 .5 .9]);
fx.mfvar_csv = {replay_case(61, @bvar.models.mfvar_csv, {Y, 2}, 1, o1), ...
    replay_case(62, @bvar.models.mfvar_csv, {Y, 2}, 1, o2)};
    % short chains of both specifications, and the untruncated phi candidate
[Y0, Y] = var_data(9, 3, 80, 6);
fx.var_sv = {replay_case(94, @bvar.models.var_sv, {Y0, Y, 2}, 1, ...
        struct('model', 'OI', 'nsim', 5, 'burnin', 3, 'draws', true)), ...
    replay_case(95, @bvar.models.var_sv, {Y0, Y, 2}, 1, struct('model', 'CS', 'nsim', 5, 'burnin', 3, 'draws', true)), ...
    replay_case(96, @bvar.models.var_sv, {Y0, Y, 2}, 1, ...
        struct('model', 'OI', 'nsim', 4, 'burnin', 2, 'phi_proposal', 'untruncated'))};
fx.var_csv = {replay_case(100, @bvar.models.var_csv, {Y0, Y, 2}, 1, ...
        struct('nsim', 6, 'burnin', 18, 'draws', true)), ...
    replay_case(101, @bvar.models.var_csv, {Y0, Y, 2}, 1, ...
        struct('nsim', 3, 'burnin', 2, 'kappa', [.1 10], 'c_reject', 2))};
end

function fx = structural_cases()
rng(6, 'twister');
n = 4;  p = 2;  nsim = 6;
fx.reduced_form = {run_case(@bvar.structural.reduced_form, ...
    {randn(nsim, n*(n-1)/2), randn(nsim, n^2*p + n), rand(nsim, n) + .5}, 2)};
fx.qr_sign = {run_case(@bvar.structural.qr_sign, {randn(5)}, 2), ...
    run_case(@bvar.structural.qr_sign, {randn(8)}, 2)};
fx.irf_redu = {run_case(@bvar.structural.irf_redu, {.2*randn(6, 3), randn(3), 6, 2}, 1), ...
    run_case(@bvar.structural.irf_redu, {.3*randn(4, 4), randn(4), 4, 4}, 1)};

    % three shocks in six variables, separated by their signs
S6 = nan(6, 3);  S6([1 2], 1) = 1;  S6([1 3], 2) = 1;  S6(2, 2) = -1;  S6(1, 3) = -1;  S6([2 3], 3) = 1;
Sbad = nan(4, 3);  Sbad([1 2], 1) = 1;  Sbad([1 2], 2) = 1;  Sbad(1, 3) = -1;  Sbad(3, 3) = 1;
[S35, R35] = tutorial_restrictions();
fx.check_separable = {run_case(@bvar.structural.check_separable, {S35, R35}, 2), ...
    run_case(@bvar.structural.check_separable, {S35}, 2), ...
    run_case(@bvar.structural.check_separable, {S6}, 2), ...
    run_case(@bvar.structural.check_separable, {Sbad}, 2)};

    % sign_restrict: three shocks in five variables and two row inequalities; candidates
    % that it accepts, that fail after flipping a column, that fail on a sign without a
    % flip, and that pass the signs and fail on a row
S5 = nan(5, 3);  S5([1 2], 1) = 1;  S5(3, 2) = -1;  S5([1 4], 3) = [1; -1];
Rrows = [1 -1 0 0 0; 0 0 1 0 -1];  Ridx = [1; 3];
found = cell(1, 4);
while any(cellfun(@isempty, found))
    L = randn(5);
    [ok, Lo] = bvar.structural.sign_restrict(L, S5, Rrows, Ridx);
    okS = bvar.structural.sign_restrict(L, S5, zeros(0, 5), zeros(0, 1));
    flipped = any(Lo(:) ~= L(:));
    if ok, j = 1; elseif okS, j = 4; elseif flipped, j = 2; else, j = 3; end
    if isempty(found{j}), found{j} = L; end
end
fx.sign_restrict = {};
for j = 1:4
    fx.sign_restrict{end+1} = run_case(@bvar.structural.sign_restrict, {found{j}, S5, Rrows, Ridx}, 2);
end
fx.sign_restrict{end+1} = run_case(@bvar.structural.sign_restrict, {found{1}, S5, zeros(0, 5), zeros(0, 1)}, 2);
    % a row of zeros fails the strict test and rejects the candidate the rows above accept
fx.sign_restrict{end+1} = run_case(@bvar.structural.sign_restrict, {found{1}, S5, [Rrows; zeros(1, 5)], [Ridx; 2]}, 2);

    % sign_assign on the replay stream: accepted candidates with one and with two ranking
    % restrictions per shock, a rejected one, and an accepted candidate under the
    % tutorial's 35-variable restrictions
Rk1 = zeros(3, 6);  Rk1(1, [4 5]) = [1 -1];
Rk2 = zeros(3, 6, 2);  Rk2(:, :, 1) = Rk1;  Rk2(2, [5 6], 2) = [1 -1];
acc1 = {};  acc2 = {};  rej = [];
while numel(acc1) < 2 || numel(acc2) < 2 || isempty(rej)
    L = randn(6);
    ok1 = bvar.structural.sign_assign(L, S6, Rk1, 1);
    ok2 = bvar.structural.sign_assign(L, S6, Rk2, 2);
    if ok1 && numel(acc1) < 2, acc1{end+1} = L; end %#ok<AGROW>
    if ok2 && numel(acc2) < 2, acc2{end+1} = L; end %#ok<AGROW>
    if ~ok1 && ~ok2 && isempty(rej), rej = L; end
end
    % candidates for the 35 variables, drawn as in the tutorial: a posterior draw of the
    % covariance under the asymmetric conjugate prior at the tutorial's hyperparameters,
    % its Cholesky factor rotated once
here = fileparts(mfilename('fullpath'));
csv = fullfile(fileparts(fileparts(here)), 'data', 'chan_matthes_yu2026_qe_svarsign', 'data35_1975.csv');
data = readmatrix(csv, 'Range', 'B35:AJ182');
Y0 = data(1:8, :);  Y = data(9:end, :);
prior = bvar.priors.acp_redu(35, 5, [.2444 .0025 1 100], bvar.priors.resid_var_ar4(Y0, Y), 1:35);
L35 = [];
for b = 1:200
    [alp, beta, sig] = bvar.samplers.acp_theta_sig(Y0, Y, 5, prior, 1000);
    [~, Sigt] = bvar.structural.reduced_form(alp, beta, sig);
    for d = 1:1000
        L = chol(squeeze(Sigt(d,:,:)), 'lower')*bvar.structural.qr_sign(randn(35));
        if bvar.structural.sign_assign(L, S35, R35, 2), L35 = L; break, end
    end
    if ~isempty(L35), break, end
end
assert(~isempty(L35), 'no accepted 35-variable candidate in 200,000 tries');
fx.sign_assign = {replay_case(31, @bvar.structural.sign_assign, {acc1{1}, S6, Rk1, 1}, 2), ...
    replay_case(32, @bvar.structural.sign_assign, {acc1{2}, S6, Rk1, 1}, 2), ...
    replay_case(33, @bvar.structural.sign_assign, {acc2{1}, S6, Rk2, 2}, 2), ...
    replay_case(34, @bvar.structural.sign_assign, {acc2{2}, S6, Rk2, 2}, 2), ...
    replay_case(35, @bvar.structural.sign_assign, {rej, S6, Rk1, 1}, 2), ...
    replay_case(36, @bvar.structural.sign_assign, {L35, S35, R35, 2}, 2)};

    % covariance paths from a full and a unit lower triangular impact matrix, and the
    % row-wise draw of B0 on the residuals of a generated VAR
hs = .3*randn(5, 3);
fx.construct_Sigt = {run_case(@bvar.structural.construct_Sigt, {hs, eye(3) + .3*randn(3)}, 1), ...
    run_case(@bvar.structural.construct_Sigt, {hs, [1 0 0; .4 1 0; -.2 .3 1]}, 1)};
Ub = randn(60, 4)*chol([1 .3 0 .1; .3 1 .2 0; 0 .2 1 .4; .1 0 .4 1]);
hb = .3*sin((1:60)'/8 + (1:4));
fx.b0_row_sampler = {replay_case(92, @bvar.structural.b0_row_sampler, ...
        {Ub, hb, eye(4) + .1*randn(4), eye(4), ones(4)}, 1), ...
    replay_case(93, @bvar.structural.b0_row_sampler, {Ub, hb, eye(4), .5*eye(4) + .1, 2*ones(4)}, 1)};
end

function [S, Rineq] = tutorial_restrictions()
% the sign and ranking restrictions of tutorials/sign_restrictions/build.m in bvar-toolkit
n = 35;
pos = {[1 10 11 12 13 14 20 21 25 26 30], [1 10 11 12 13 14 20 21 25 26 30], ...
       [1 10 11 12 13 14 20 21 25 26 30 34], [19 25 26 27 28 29 30 31 32], ...
       [1 7 9 10 11 12 13 14 25 26 30], [1 2 4 15 16 17], [1 19], 1};
neg = {19, [19 34], 19, [1 10 11 12 13 14 18 20 21 34], [8 19], ...
       [10 11 12 13 14 19], [10 11 12 13 14 15], [10 11 12 13 14 15 19]};
m = numel(pos);
S = nan(n, m);
for i = 1:m, S(pos{i},i) = 1;  S(neg{i},i) = -1; end
Rineq = zeros(m, n, 2);
Rineq(1,[1 4],1) = [-1 1];  Rineq(2,[1 4],1) = [1 -1];  Rineq(3,[1 4],1) = [1 -1];
Rineq(1,[1 7],2) = [-1 1];  Rineq(2,[1 7],2) = [-1 1];  Rineq(3,[1 7],2) = [-1 1];
Rineq(5,[1 7],2) = [1 -1];
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
    % one draw of each covariance specification: with the realized values, with a missing
    % row of them, and without them
n = 3;  p = 2;
A = [.1*randn(1, n); .3*eye(n) + .05*randn(n); .1*eye(n)];
cfg = struct('ylag', randn(n, p), 'H', 4, 'yobs', randn(4, n));
cfg2 = cfg;  cfg2.yobs(3, 2) = NaN;
cfg3 = struct('ylag', randn(n, p), 'H', 3);
Sg = [1 .3 .1; .3 1 .2; .1 .2 1];
dg = struct('A', A, 'Sig', Sg);
dc = struct('A', A, 'Sig', Sg, 'h_T', .3, 'phi', .9, 'sigh2', .1);
do = struct('A', A, 'impact', eye(n) + .2*randn(n), 'h_T', [.1 -.2 .3], 'phi', [.9 .8 .95], ...
    'sig2', [.1 .2 .05]);
fx.simulate = {run_case(@bvar.forecast.simulate, {'gauss', dg, cfg}, 4), ...
    replay_case(102, @bvar.forecast.simulate, {'csv', dc, cfg}, 4), ...
    replay_case(103, @bvar.forecast.simulate, {'oisv', do, cfg2}, 4), ...
    replay_case(104, @bvar.forecast.simulate, {'oisv', do, cfg3}, 4)};
fx.mixquantile = {run_case(@bvar.forecast.mixquantile, {randn(50, 1), .5 + rand(50, 1), [.05 .5 .95]}, 1), ...
    run_case(@bvar.forecast.mixquantile, {[0 3], [1 .2], [.1 .3; .7 .99]}, 1)};

    % one accumulation row of forecast scores, and the tables of both schemes from stacks
    % of such rows
rng(22, 'twister');
nf = 4;  nsim = 8;
scores = @() [randn(nsim, nf), -1 - rand(nsim, nf), -4 - rand(nsim, 1)];
rows0 = zeros(10, 3*nf + 1);  rows1 = zeros(9, 3*nf + 1);  rows4 = zeros(6, 3*nf + 1);
for i = 1:10, rows0(i, :) = bvar.forecast.tables('accum_row', scores(), randn(1, nf)); end
for i = 1:9, rows1(i, :) = bvar.forecast.tables('accum_row', scores(), randn(1, nf)); end
for i = 1:6, rows4(i, :) = bvar.forecast.tables('accum_row', scores(), randn(1, nf)); end
fx.tables = {run_case(@bvar.forecast.tables, {'accum_row', scores(), randn(1, nf)}, 1), ...
    run_case(@bvar.forecast.tables, {'springer', rows0, rows1, []}, 1), ...
    run_case(@bvar.forecast.tables, {'springer', rows0, rows1, [1; 3]}, 1), ...
    run_case(@bvar.forecast.tables, {'mahp', rows1, rows4}, 1)};

    % a real-time panel of four series, one of each frequency code, three of them in growth
    % rates, at two vintages
Lq = 20;  T0 = 3;  nv = Lq - T0 + 1;
lvl = @(L) 100*exp(cumsum(.01 + .005*randn(L, 1)));
rt = struct('var1', lvl(Lq).*(1 + .001*randn(Lq, nv)), 'var2', lvl(3*Lq).*(1 + .001*randn(3*Lq, nv)), ...
    'var3', lvl(3*Lq).*(1 + .001*randn(3*Lq, 3*nv)));
nonrev = struct('var4', 5 + randn(3*Lq + 1, 1));
fx.realtime_loaddata = {run_case(@bvar.forecast.realtime_loaddata, {rt, nonrev, 12, T0, [5; 5; 5; 1], [1; 2; 3; 4]}, 2), ...
    run_case(@bvar.forecast.realtime_loaddata, {rt, nonrev, 16, T0, [5; 5; 5; 1], [1; 2; 3; 4]}, 2)};

    % one draw of each branch of the iterated forecasts: with every horizon evaluated, with
    % the latest observation missing and the last horizon past the sample, and with the last
    % horizon at the end of the sample
n = 3;  p = 2;  Tt = 30;
A = [.1*randn(1, n); .3*eye(n) + .05*randn(n); .1*eye(n) + .02*randn(n)];
Sig = [1 .3 -.1; .3 1 .2; -.1 .2 1];
cfg = struct('shortYt', randn(Tt, n), 'data_tpk', randn(5, n), 'is_last_miss', false, 'p', p, 't', 20, 'T', 30);
cfgm = cfg;  cfgm.is_last_miss = true;  cfgm.t = 29;
dg = struct('A', A, 'CSig', chol(Sig, 'lower'), 'dSig', diag(Sig)');
dgs = struct('A', A, 'CSig', sparse(1:n, 1:n, sqrt([1.1 .9 1.3])), 'dSig', [1.1 .9 1.3]);
dc = struct('A', A, 'CSig', chol(Sig, 'lower'), 'Sig', Sig, 'h', .2*randn(Tt, 1), 'rho', .9, 'sigh2', .1);
dt = dc;  dt.nu = 7;
dm = dt;  dm.psi = .3;  dm.Hpsi = speye(Tt) + .3*sparse(2:Tt, 1:Tt-1, 1, Tt, Tt);
cfgz = cfg;  cfgz.Z = [ones(Tt, 1), randn(Tt, n*p)];
cfgzm = cfgm;  cfgzm.Z = cfgz.Z;
Yfull = randn(40, n);
dsv = struct('alp', .3*randn(n*(n-1)/2, 1), 'beta', A(:), 'h_T', .2*randn(n, 1), 'Sigh', [.05; .1; .02]);
cs1 = struct('Yt', Yfull(1:20, :), 'Y', Yfull, 'p', p, 't', 20, 'T', 40);
cs2 = struct('Yt', Yfull(1:38, :), 'Y', Yfull, 'p', p, 't', 38, 'T', 40);
cs3 = cs2;  cs3.t = 36;
cfge = cfg;  cfge.t = 28;
fx.iterate = {replay_case(246, @bvar.forecast.iterate, {'mahp_sv', dsv, cs1}, 1), ...
    replay_case(247, @bvar.forecast.iterate, {'mahp_sv', dsv, cs2}, 1), ...
    replay_case(248, @bvar.forecast.iterate, {'springer_gauss', dg, cfg}, 1), ...
    replay_case(249, @bvar.forecast.iterate, {'springer_gauss', dgs, cfgm}, 1), ...
    replay_case(250, @bvar.forecast.iterate, {'springer_csv', dc, cfg}, 1), ...
    replay_case(251, @bvar.forecast.iterate, {'springer_csv', dc, cfgm}, 1), ...
    replay_case(252, @bvar.forecast.iterate, {'springer_csv_t', dt, cfg}, 1), ...
    replay_case(253, @bvar.forecast.iterate, {'springer_csv_t', dt, cfgm}, 1), ...
    replay_case(254, @bvar.forecast.iterate, {'springer_csv_t_ma', dm, cfgz}, 1), ...
    replay_case(255, @bvar.forecast.iterate, {'springer_csv_t_ma', dm, cfgzm}, 1), ...
    replay_case(256, @bvar.forecast.iterate, {'mahp_sv', dsv, cs3}, 1), ...
    replay_case(257, @bvar.forecast.iterate, {'springer_csv_t', dt, cfge}, 1)};
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
g1 = @(x) x*x*x - 2*x - 5;
g2 = @(x) cos(x) - x;
g3 = @(x) x*x - 1e-6;
fx.fzero = {opt_case('fzero', 'g1', g1, {[2 3]}), opt_case('fzero', 'g2', g2, {[0 1]}), ...
    opt_case('fzero', 'g3', g3, {[0 1]}), opt_case('fzero', 'g3', g3, {[-.3 -.0005]})};
    % the fits of the Statistics and Machine Learning Toolbox the marginal likelihoods use
rng(10, 'twister');
fx.gamfit = {run_case(@gamfit, {gamrnd(2.5, .3, 200, 1)}, 1), run_case(@gamfit, {gamrnd(40, .01, 50, 1)}, 1), ...
    run_case(@gamfit, {exp(.3*randn(100, 1))*.04}, 1)};
fx.betafit = {run_case(@betafit, {betarnd(2, 30, 200, 1)}, 1), run_case(@betafit, {betarnd(5, 5, 60, 1)}, 1)};
    % the sum and gammaln the two fits use, which the Python versions must equal exactly:
    % vectors whose entries differ in size by orders of magnitude, so that the order of the
    % additions shows in the last digit, and points in every branch of gammaln
rng(11, 'twister');
fx.sum = {};
for m = [1 3 4 5 8 11 50 203 1000]
    fx.sum{end+1} = run_case(@sum, {randn(m, 1).*exp(3*randn(m, 1))}, 1);
end
xg = [0 1e-20 eps .01 .3 .5 .6 .6796875 .9 1 1.2 1.5 2 2.5 3.7 4 5 11.5 12 13 40 99 100 ...
    100.5 800.25 1e5 1e20 1e80 1e300, 20*rand(1, 200), exp(14*rand(1, 100))];
fx.gammaln = {run_case(@gammaln, {xg}, 1)};
    % linspace, on the grids of the marginal likelihoods and on the special cases of its
    % arithmetic
fx.linspace = {run_case(@linspace, {-.999, .999, 700}, 1), run_case(@linspace, {-.99, .99, 300}, 1), ...
    run_case(@linspace, {-.999, .999, 299}, 1), run_case(@linspace, {2, 50, 700}, 1), ...
    run_case(@linspace, {2, 30, 300}, 1), run_case(@linspace, {0, 1, 7}, 1), ...
    run_case(@linspace, {-1, 1, 2}, 1), run_case(@linspace, {-1, 1, 3}, 1), run_case(@linspace, {1, 1, 4}, 1)};
    % fminunc, which the Python port replaces by a Nelder-Mead search followed by the same
    % finite-difference second derivative
u1 = @(x) .5*(x - .25)^2/.01 + .5*log(1 + x^2);
u2 = @(x) 3*log(1 + (x - .4)^2/.2) + .5*x^2;
fx.fminunc = {unc_case('u1', u1, .5), unc_case('u1', u1, -.4), unc_case('u2', u2, 1.5)};
end

function c = unc_case(name, f, x0)
[x, fval, ~, ~, ~, hess] = fminunc(f, x0, optimset('Display', 'off', 'LargeScale', 'off'));
c = pack({x0}, {x, fval, hess});
c.name = name;
end

function fx = replay_cases()
% the functions of replay/ themselves, called by name so that the replay versions answer
named = @(name) @(varargin) feval(name, varargin{:});
Tau = [2 .5 0 .1; .5 1 .2 0; 0 .2 1.5 .3; .1 0 .3 1];
fx.randn = {replay_case(71, named('randn'), {3, 2}, 1)};
fx.rand = {replay_case(72, named('rand'), {2, 3}, 1)};
fx.gamrnd = {replay_case(73, named('gamrnd'), {[.5; 2; 7], 1.5}, 1)};
fx.betarnd = {replay_case(74, named('betarnd'), {[.5 2], [3 4]}, 1)};
fx.unidrnd = {replay_case(75, named('unidrnd'), {6, 2, 3}, 1)};
fx.randi = {replay_case(76, named('randi'), {9}, 1), replay_case(77, named('randi'), {9, 2, 3}, 1)};
fx.randperm = {replay_case(78, named('randperm'), {6}, 1)};
fx.iwishrnd = {replay_case(79, named('iwishrnd'), {Tau, 9}, 1), ...
    replay_case(80, named('iwishrnd'), {2.5, 4}, 1)};
end

function fx = drivers_cases(matlab_repo, root)
% run_ml of the Chan (2023) package in bvar-toolkit, which tutorials/sv_specification/ml_varsv.py
% ports, on the tutorial's data with short chains and the replay stream, on which gigrnd returns
% the mode of its distribution in both languages. A case keeps the posterior means of the stored
% draws.
addpath(fullfile(matlab_repo, 'replications', 'chan2023_joe_mlvarsv'));
tbl = readtable(fullfile(root, 'tutorials', 'sv_specification', 'macro5_Q.csv'));
data = tbl{:, {'UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1'}};
spec = {'VAR-NCP', false, 1; 'VAR-CSV', false, 1; 'VAR-SV', false, 1; 'VAR-SV', true, 1; ...
        'VAR-FSV', false, 1; 'VAR-FSV', false, 2; 'VAR-SVO', false, 1};
flds = {'store_h', 'store_hpara', 'store_kappa', 'store_beta', 'store_l', 'store_o', 'store_po'};
fx.run_ml = {};
for i = 1:size(spec, 1)
    c = struct('model', spec{i, 1}, 'sym', spec{i, 2}, 'r', spec{i, 3}, 'seed', 140 + i, ...
        'nsim', 30, 'burnin', 20, 'M', 100);
    replay_start(c.seed);
    [~, out] = evalc(['run_ml(c.model, false, c.sym, c.nsim, c.burnin, [], [], ''data'', ' ...
        'data, ''r'', c.r, ''M'', c.M)']);
    c.count = replay_stop();
    c.lml = out.lml;
    if ~isempty(out.lmlstd), c.lmlstd = out.lmlstd; end
    for f = flds
        if isfield(out, f{1}), c.(['mean_' f{1}]) = squeeze(mean(out.(f{1}), 1)); end
    end
    fx.run_ml{end+1} = c;
end
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

function c = replay_case(seed, fn, in, nout, opts)
% the outputs of fn(in{:}, name-value options) on the replay stream of the seed
if nargin < 5, opts = struct(); end
nv = reshape([fieldnames(opts)'; struct2cell(opts)'], 1, []);
replay_start(seed);
cleanup = onCleanup(@() replay_stop());
out = cell(1, nout);
[out{:}] = fn(in{:}, nv{:});
c = pack(in, out);
c.opts = opts;
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

function [Y, M, z, sig2, A, Sig, h, Yfull] = mf_problem(seed, T)
% three monthly series from a stable VAR(2), the third observed only through its quarterly
% aggregate, with one monthly value missing inside the sample and one in the initial
% conditions; the restrictions, the Minnesota scales and parameters near the truth
rng(seed, 'twister');
n = 3;  p = 2;
B1 = .5*eye(n) + .05*randn(n);  B2 = .1*eye(n);  b0 = .1*ones(n, 1);
y = zeros(T, n);
for t = 3:T
    y(t, :) = (b0 + B1*y(t-1, :)' + B2*y(t-2, :)' + .5*randn(n, 1))';
end
X = y;
X(:, 3) = NaN;
for t = 6:3:T
    X(t, 3) = [1 2 3 2 1]/3*y(t:-1:t-4, 3);
end
X(17, 1) = NaN;  X(1, 2) = NaN;
[M, z, Y] = bvar.util.mm_constraint(X, [false false true]);
sig2 = [NaN; NaN; .1];
A = [b0'; B1'; B2'];
Sig = .25*eye(n) + .02;
h = .3*sin((1:T-p)'/7);
Yfull = y;
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
