% Define paths
clear; close all; clc;


load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/generated-Shuffle/Generated/SumTimeSeries.mat');
y = ts_sum


fs = 1000;
MAX_OSC = 6;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[minAIC,K] = min(osc_AIC);
osc_a = osc_param(K,1:K);
osc_f = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2 = osc_param(K,3*K+1);
[hess,grad,mll] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);
fprintf('The number of oscillators is K=%d.\n',K);
fprintf('The periods of K oscillators are:\n');
for k=1:K
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',1./osc_f(k),1./(osc_f(k)+1.96*sqrt(cov_est(K+k,K+k))),1./(osc_f(k)-1.96*sqrt(cov_est(K+k,K+k))));
end
osc_plot(osc_mean,osc_cov,fs,K)
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K)
osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2)
%% 
clear; close all; clc;

% Load the cell array containing the filtered data for Bandpass 200, Part 1.
% This file must contain the variable 'Filtered_200_Part1' (a cell array with 503 trials).
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part4.mat');  

fs = 1000;      % Sampling frequency
MAX_OSC = 4;    % Maximum number of oscillators to use in the model
numTrials = 10;  % Process the first 100 trials

% Preallocate a cell array to store the frequency vector (osc_param(K, K+1:2*K)) for each trial.
freqResults = cell(numTrials, 1);

for trial = 1:numTrials
    % Extract the trial data (each trial may have a different length)
    y = Filtered_200_Part4{trial};
    
    % Run the oscillation decomposition model on the trial data.
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    
    % Determine the optimal model order K based on minimum AIC.
    [~, K] = min(osc_AIC);
    
    % Extract the frequency parameters from osc_param:
    % For the optimal order K, extract the elements from column (K+1) to (2*K).
    freq_vector = osc_param(K, K+1:2*K);
    
    % Store the frequency vector for the current trial.
    freqResults{trial} = freq_vector;
end

% Save the frequency results for the first 100 trials into a .mat file.
% The file will be named 'Filtered_200_Part1_F.mat'.
save('Filtered_200_Part14_F_10.mat', 'freqResults_F200_P1_10');

%%


clear; close all; clc;

% Load the cell array containing the filtered data for Bandpass 200, Part 1.
% This file should contain the variable 'Filtered_200_Part1' (a cell array with 503 trials).
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part1.mat');  

fs = 1000;      % Sampling frequency
MAX_OSC = 5;    % Maximum number of oscillators to consider in the model


% Process the second 100 trials: trials 101 to 200.
startTrial = 451;
endTrial = 500;
numTrials = endTrial - startTrial + 1;  % 100 trials

% Preallocate cell array for frequency results and vector for optimal K values.
freqResults_F_200_Part1_500 = cell(numTrials, 1);
K_F_200_Part4_110 = zeros(numTrials, 1);

for i = 1:numTrials
    trialIdx = startTrial + i - 1;
    y = Filtered_200_Part1{trialIdx};  % Get the data for the current trial
    
    % Run the oscillation decomposition model on the trial data.
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    
    % Determine the optimal model order K based on the minimum AIC.
    [~, K] = min(osc_AIC);
    K_F_200_Part4_110(i) = K;  % Save the optimal K value for this trial
    
    % Extract the frequency parameters from osc_param for the optimal K.
    % The frequency vector is located in columns (K+1) to (2*K).
    freq_vector = osc_param(K, K+1:2*K);
    
    % Save the frequency vector for the current trial.
    freqResults_F_200_Part1_500{i} = freq_vector;
end

% Save the frequency results to a .mat file with the specified variable name.
save('Filtered_200_Part1_F_500.mat', 'freqResults_F_200_Part1_500');

% Save the optimal K values for each trial to a separate .mat file.
%save('Filtered_200_Part3_K_51U70.mat', 'K_F_200_Part3_51U70');








%%

%load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
% Make sure the .mat file and variable name are correct
% Define file paths for y_1 to y_4 and raw data for y_5
y_1_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/generated-Shuffle/Generated/SumTimeSeries.mat'


y_2_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_2.mat' 



y_3_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_3.mat'

y_4_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_4.mat'
%y_5_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/same_Variable_Name/Cue1LFP_Filtered_400.mat';

% Load data for y_1 to y_4
y_1 = load(y_1_path);
% y_2 = load(y_2_path);
% y_3 = load(y_3_path);
% y_4 = load(y_4_path);
%y_5 = load(y_5_path);
% Ensure proper field extraction
y_1 = get_single_variable(y_1);
% y_2 = get_single_variable(y_2);
% y_3 = get_single_variable(y_3);
% y_4 = get_single_variable(y_4);
%y_5 = y_5.Cue1LFP_Filtered(313,:);
% Raw data for y_5
%y_5 = Cue1LFP(313, :); % Replace 'Cue1LFP' with the actual variable containing raw data

% Collect all datasets into a cell array
datasets = {y_1};
dataset_names = {'y_1'};

% Define save path for results
save_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Shuffle';
mkdir(save_path); % Create directory if it doesn't exist

% Sampling frequency and maximum oscillators
fs = 1000;       % Sampling frequency
MAX_OSC = 5;     % Maximum number of oscillators
indices_to_process =  [1];

% Loop through each dataset
for i = indices_to_process
    y = datasets{i};          % Current dataset
    dataset_name = dataset_names{i}; % Dataset name for saving files

    fprintf('Processing dataset: %s\n', dataset_name);

    % Perform osc decomposition
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    [minAIC, K] = min(osc_AIC);
    osc_a = osc_param(K, 1:K);
    osc_f = osc_param(K, K+1:2*K);
    osc_sigma2 = osc_param(K, 2*K+1:3*K);
    osc_tau2 = osc_param(K, 3*K+1);

    % Save plots
    osc_plot(osc_mean, osc_cov, fs, K);
    saveas(gcf, sprintf('%s/%s_Oscillation.png', save_path, dataset_name));
    close;

    osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
    saveas(gcf, sprintf('%s/%s_Phase.png', save_path, dataset_name));
    close;

    osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
    saveas(gcf, sprintf('%s/%s_Spectrum.png', save_path, dataset_name));
    close;

    % Save osc_param as .mat file
    save(sprintf('%s/%s_osc_param.mat', save_path, dataset_name), 'osc_param');
end

fprintf('Processing complete. Results saved to %s.\n', save_path);



%%

load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
% Make sure the .mat file and variable name are correct
base_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Filtered_Data_Windowed_BandPass 50,...400';
save_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Analysis_Results';
mkdir(save_path); % Create save directory if it doesn't exist

% Define parameters
fs = 1000;       % Sampling frequency
MAX_OSC = 5;     % Maximum number of oscillators
selected_trials = [313]; % List of selected trials

% Get all subfolders in the base path
subfolders = dir(base_path);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

% Loop through each subfolder
for i = 1:length(subfolders)
    folder_name = subfolders(i).name;
    folder_path = fullfile(base_path, folder_name);

    % Parse trial and filter specification from folder name
    tokens = regexp(folder_name, 'Trials_(\d+)_filtered_(\d+)', 'tokens');
    if isempty(tokens)
        warning('Folder name "%s" does not match the expected pattern. Skipping.', folder_name);
        continue;
    end
    trial_num = str2double(tokens{1}{1});
    filter_spec = tokens{1}{2};

    fprintf('Processing folder: %s (Trial: %d, Filter: %s)\n', folder_name, trial_num, filter_spec);

    % Load all 4 components in the folder
    component_files = dir(fullfile(folder_path, '*.mat'));
    components = cell(1, 4);
    for j = 1:4
        component_file = fullfile(folder_path, sprintf('part_%d.mat', j));
        if isfile(component_file)
            data = load(component_file);
            field_names = fieldnames(data);
            components{j} = data.(field_names{1}); % Assuming the .mat file has one variable
        else
            warning('File not found: %s', component_file);
            components{j} = [];
        end
    end

    % Run the code for each component
    for j = 1:4
        if isempty(components{j})
            continue;
        end
        y = components{j};

        % Perform osc decomposition
        [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
        [minAIC, K] = min(osc_AIC);
        osc_a = osc_param(K, 1:K);
        osc_f = osc_param(K, K+1:2*K);
        osc_sigma2 = osc_param(K, 2*K+1:3*K);
        osc_tau2 = osc_param(K, 3*K+1);

        % Save plots
        osc_plot(osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/%s_Component_%d_Oscillation.png', save_path, folder_name, j));
        close;

        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/%s_Component_%d_Phase.png', save_path, folder_name, j));
        close;

        osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
        saveas(gcf, sprintf('%s/%s_Component_%d_Spectrum.png', save_path, folder_name, j));
        close;

        % Save osc_param as .mat file
        save(sprintf('%s/%s_Component_%d_osc_param.mat', save_path, folder_name, j), 'osc_param');
    end

    % Process the raw trial data for the selected trial
    if ismember(trial_num, selected_trials)
        fprintf('Processing raw data for Trial %d...\n', trial_num);
        y = Cue1LFP(trial_num, :);

        % Perform osc decomposition
        [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
        [minAIC, K] = min(osc_AIC);
        osc_a = osc_param(K, 1:K);
        osc_f = osc_param(K, K+1:2*K);
        osc_sigma2 = osc_param(K, 2*K+1:3*K);
        osc_tau2 = osc_param(K, 3*K+1);

        % Save plots
        osc_plot(osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Oscillation.png', save_path, trial_num));
        close;

        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Phase.png', save_path, trial_num));
        close;

        osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Spectrum.png', save_path, trial_num));
        close;

        % Save osc_param as .mat file
        save(sprintf('%s/Trial_%d_Raw_osc_param.mat', save_path, trial_num), 'osc_param');
    end
end

fprintf('Processing complete. Results saved to %s.\n', save_path);

%%
clear all;
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/generated-Shuffle/Generated/TimeSeries5.mat');


y = ts;
% y_1 = average_part1;
%  load('CanadianLynxData.mat');
%  y_2 = lynx;
% y = y_1(1:114).*10000000 + y_2;
%  y = log(y_2.*100);
% 
 %y = (y-mean(y))/1000;
%y = log(y);
%y = channelData(1:1800);




fs = 1000;
MAX_OSC = 5;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[minAIC,K] = min(osc_AIC);
osc_a = osc_param(K,1:K);
osc_f = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2 = osc_param(K,3*K+1);
[hess,grad,mll] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);
fprintf('The number of oscillators is K=%d.\n',K);
fprintf('The periods of K oscillators are:\n');
for k=1:K
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',1./osc_f(k),1./(osc_f(k)+1.96*sqrt(cov_est(K+k,K+k))),1./(osc_f(k)-1.96*sqrt(cov_est(K+k,K+k))));
end
osc_plot(osc_mean,osc_cov,fs,K)
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K)
osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2)

%%


% Compute the sum of all oscillator mean components across K
% Assuming osc_mean has dimensions [2*K-1 x T x K]
% Adjust indexing based on your actual data structure

% Initialize the sum signal
sum_osc_mean = zeros(1, size(osc_mean, 2));
T = size(osc_mean,2);
% Sum across all K oscillators
for k = 1:K
    % Assuming that the mean of each oscillator is stored in osc_mean(2*k-1, :, k)
    sum_osc_mean = sum_osc_mean + osc_mean(2*k-1, :, k);
end
% Plot the sum of all oscillator components
figure;
plot((1:T)/fs, sum_osc_mean, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Sum of All Oscillator Components');
grid on;
set(gca, 'FontSize', 12);
figure;
hold on;
plot((1:T)/fs, y, 'k-', 'DisplayName', 'Original Signal');
%plot((1:T)/fs, sum_osc_mean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sum of Oscillators');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal vs. Sum of Oscillator Components');
legend;
grid on;
set(gca, 'FontSize', 12);
hold off;
%%
figure;

plot((1:T)/fs, y - sum_osc_mean, 'b-', 'LineWidth', 1.5);
%%
% Initialize arrays to store the mean phases
mean_phase = zeros(1, T);

% Compute the mean phase using circular statistics
for t = 1:T
    % Extract phases of all K oscillators at time t
    phases = osc_phase(:, t, K); % Adjust indexing based on your data structure
    
    % Convert phases to unit vectors
    unit_vectors = exp(1i * phases);
    
    % Compute the mean resultant vector
    mean_vector = mean(unit_vectors);
    
    % Compute the mean phase angle
    mean_phase(t) = angle(mean_vector);
end
% Initialize arrays to store the summed phase vectors
sum_phase_vector = zeros(1, T);

% Compute the sum of phase vectors
for t = 1:T
    % Extract phases of all K oscillators at time t
    phases = osc_phase(:, t, K); % Adjust indexing based on your data structure
    
    % Convert phases to unit vectors and sum them
    sum_vector = sum(exp(1i * phases));
    
    % Compute the angle of the summed vector
    sum_phase_vector(t) = angle(sum_vector);
end

% Plot the summed phase vectors
figure;
plot((1:T)/fs, sum_phase_vector, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Summed Phase (radians)');
title('Summed Phase Across All Oscillators');
grid on;
set(gca, 'FontSize', 12);
ylim([-pi, pi]);
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});
%%

% Hilbert for Phase
% Assuming y is your time series and fs is the sampling frequency
y = y(:); % Convert to column vector if it's not already
% Compute the analytic signal
analytic_signal = hilbert(y); % Use y if not filtering
% Calculate instantaneous phase in radians
inst_phase = angle(analytic_signal);
% Unwrap the instantaneous phase
inst_phase_unwrapped = unwrap(inst_phase);
% Define time vector based on sampling frequency
T = length(y);
t = (0:T-1)' / fs; % Time in seconds

% Create a figure with two subplots
figure;

% Plot the original or filtered signal
subplot(2,1,1);
plot(t, y, 'b'); % Use y instead of y_filtered if not filtering
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');
grid on;
set(gca, 'FontSize', 12);

% Plot the instantaneous phase
subplot(2,1,2);
plot(t, inst_phase_unwrapped, 'r');
xlabel('Time (s)');
ylabel('Phase (radians)');
title('Instantaneous Phase (Unwrapped)');
grid on;
set(gca, 'FontSize', 12);
ylim([-pi, pi]);
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});

%%
% Helper function to extract the single variable from the loaded .mat file
function data = get_single_variable(loaded_data)
    fields = fieldnames(loaded_data);
    if length(fields) == 1
        data = loaded_data.(fields{1});
    else
        error('Loaded .mat file contains multiple variables. Ensure the file has a single variable.');
    end
end
