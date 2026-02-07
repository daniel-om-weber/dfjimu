% export_reference.m — Master export script
% Runs MEKF and GN optimizer on all datasets and saves reference outputs.
%
% Usage: Run from the matlab_implementation/ directory:
%   cd matlab_implementation
%   run('../tests/matlab_export/export_reference.m')
%
% Or add paths manually and run from any directory.

clear; clc;

%% Setup paths
% Determine script location
script_dir = fileparts(mfilename('fullpath'));
matlab_dir = fullfile(script_dir, '..', '..', 'matlab_implementation');
output_dir = fullfile(script_dir, '..', 'reference_outputs');

% Add MATLAB library paths
addpath(genpath(fullfile(matlab_dir, 'lib')));
addpath(genpath(fullfile(matlab_dir, 'OPT')));
addpath(genpath(fullfile(matlab_dir, 'MEKF')));
addpath(genpath(fullfile(matlab_dir, 'data')));

% Also add the wrapper directory
addpath(script_dir);

% Create output directory if needed
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Dataset list (all 15 datasets)
datasets = {
    'data_1D_01', 'data_1D_02', 'data_1D_03', 'data_1D_04', 'data_1D_05', ...
    'data_2D_01', 'data_2D_02', 'data_2D_03', 'data_2D_05', 'data_2D_07', ...
    'data_3D_01', 'data_3D_02', 'data_3D_03', 'data_3D_04', 'data_3D_05'  ...
};

%% Required globals for library functions (integrateGyr, approxDerivative, calculateK)
global Fs T N g gn;
Fs = 50;
T = 1/Fs;
g = 9.82;
gn = [0 0 -g]';

%% Process each dataset
for d = 1:length(datasets)
    dataset_name = datasets{d};
    fprintf('\n=== Processing %s (%d/%d) ===\n', dataset_name, d, length(datasets));

    try
        %% Load data (matches run.m lines 11-21)
        load([dataset_name '.mat']);

        % Changing namings and dimensions (matches run.m lines 16-21)
        r1 = data.r_12'; r1 = -r1;
        r2 = data.r_21'; r2 = -r2;
        acc   = data.sensorData(:,1:3)';
        gyr   = data.sensorData(:,4:6)';
        acc_2 = data.sensorData(:,7:9)';
        gyr_2 = data.sensorData(:,10:12)';

        N = size(gyr, 2);

        %% Covariances (matches run.m lines 38-44)
        noise = gyr(1:3, 1:40); sdNoise_gyr1 = std(noise');
        noise = gyr_2(1:3, 1:40); sdNoise_gyr2 = std(noise');
        cov_w = eye(6);
        cov_w(1,1) = sdNoise_gyr1(1); cov_w(2,2) = sdNoise_gyr1(2); cov_w(3,3) = sdNoise_gyr1(3);
        cov_w(4,4) = sdNoise_gyr2(1); cov_w(5,5) = sdNoise_gyr2(2); cov_w(6,6) = sdNoise_gyr2(3);
        cov_i = eye(3)*0.35^2;
        cov_lnk = eye(3);

        %% Preprocessing (matches run.m lines 47-50)
        dgyr   = [approxDerivative(gyr(1,:)); approxDerivative(gyr(2,:)); approxDerivative(gyr(3,:))];
        dgyr_2 = [approxDerivative(gyr_2(1,:)); approxDerivative(gyr_2(2,:)); approxDerivative(gyr_2(3,:))];
        [C1, D1] = calcAccatCenter(gyr, dgyr, acc, (r1)');
        [C2, D2] = calcAccatCenter(gyr_2, dgyr_2, acc_2, (r2)');

        q_init = [1 0 0 0];

        %% Run MEKF
        fprintf('  Running MEKF...\n');
        [mekf_q_s1, mekf_q_s2] = wrap_mekf(gyr, gyr_2, C1, C2, Fs, q_init, cov_w, cov_i);
        fprintf('  MEKF done. Output size: [%d x %d]\n', size(mekf_q_s1));

        %% Run GN optimizer
        fprintf('  Running GN optimizer (10 iterations)...\n');
        [opt_q_s1, opt_q_s2] = wrap_opt(gyr, gyr_2, C1, C2, Fs, q_init, cov_w, cov_i, cov_lnk, 10);
        fprintf('  GN done. Output size: [%d x %d]\n', size(opt_q_s1));

        %% Save reference outputs
        out_file = fullfile(output_dir, ['ref_' dataset_name '.mat']);
        save(out_file, ...
            'mekf_q_s1', 'mekf_q_s2', ...
            'opt_q_s1', 'opt_q_s2', ...
            'C1', 'C2', ...
            'cov_w', 'cov_i', 'cov_lnk', ...
            'r1', 'r2', ...
            'gyr', 'gyr_2', 'acc', 'acc_2', ...
            '-v7');  % v7 for scipy.io.loadmat compatibility

        fprintf('  Saved: %s\n', out_file);

    catch ME
        fprintf('  ERROR processing %s: %s\n', dataset_name, ME.message);
        fprintf('  Stack: %s\n', ME.getReport());
    end
end

fprintf('\n=== Export complete ===\n');
fprintf('Reference files saved to: %s\n', output_dir);
