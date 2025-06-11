





Y1= Y - mean(Y, 2);


%%
load('NorthSouthSunspotData');
%%
%Y = selected_trials
Y = Y - mean(Y, 2);
%Y = Y-mean(Y,2)*ones(1,size(Y,2));
fs = 1000;
% J = size(Y,1);
% T = size(Y,2);

MAX_OSC = 5;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(Y,fs,MAX_OSC);
[minAIC,K] = min(osc_AIC);
osc_a = osc_param(K,1:K);
osc_f = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_c = osc_param(K,3*K+1:(2*J+1)*K);
osc_tau2 = osc_param(K,(2*J+1)*K+1);
hess = osc_ll_hess(Y,fs,osc_param(K,1:(2*J+1)*K+1));
cov_est = inv(hess);
fprintf('The number of oscillators is K=%d.\n',K);
fprintf('The periods of K oscillators are:\n');
for k=1:K
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',1./osc_f(k),1./(osc_f(k)+1.96*sqrt(cov_est(K+k,K+k))),1./(osc_f(k)-1.96*sqrt(cov_est(K+k,K+k))));
end
fprintf('The phase differences for K oscillators correspond to:\n');
for k=1:K
    phase_diff = atan2(osc_c(2*k),osc_c(2*k-1));
    tmp = [-osc_c(2*k); osc_c(2*k-1)]/(osc_c(2*k-1)^2+osc_c(2*k)^2);
    phase_var_est = tmp'*cov_est(3*K+2*k-1:3*K+2*k,3*K+2*k-1:3*K+2*k)*tmp;
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',phase_diff/2/pi/osc_f(k),(phase_diff-1.96*sqrt(phase_var_est))/2/pi/osc_f(k),(phase_diff+1.96*sqrt(phase_var_est))/2/pi/osc_f(k));
end
osc_plot(osc_mean,osc_cov,fs,K);
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K);
osc_spectrum_plot(Y,fs,osc_a,osc_f,osc_sigma2,osc_tau2,osc_c);

%%

osc_plot(osc_mean,osc_cov,fs,K);
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K);
osc_spectrum_plot(Y,fs,osc_a,osc_f,osc_sigma2,osc_tau2,osc_c);



%%

% Parameters
N = 4;                      % Number of trials you want
offset = 0;                 % Use offset=0 for first N, offset=1 for second N, etc.

% Load the data
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part3.mat');

% Compute length of each trial
trial_lengths = cellfun(@length, Filtered_200_Part3);

% Identify all unique lengths
unique_lengths = unique(trial_lengths);

% Search for a group with at least (offset+1)*N trials of the same length
valid_group_found = false;
for i = 1:length(unique_lengths)
    len = unique_lengths(i);
    idx = find(trial_lengths == len);
    if numel(idx) >= (offset + 1) * N
        selected_indices = idx((offset*N + 1):(offset + 1)*N);
        valid_group_found = true;
        break
    end
end

if ~valid_group_found
    error('No group with at least %d trials of the same length found (offset=%d).', (offset+1)*N, offset);
end

% Extract and convert to numeric matrix
selected_trials = Filtered_200_Part3(selected_indices);
Y = cell2mat(selected_trials);  % size: L x N (if trials are column vectors)

% Report
fprintf('Selected trial indices with same length: %s\n', mat2str(selected_indices));
fprintf('Each trial has length: %d\n', len);



save_path = '/Users/aliakbarmahmoodzadeh/Desktop/TrialsForColab/';  % Desired output folder

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

for i = 1:N
    y = selected_trials{i};
    filename = sprintf('trial_%02d.mat', i);
    fullpath = fullfile(save_path, filename);
    save(fullpath, 'y');
end

fprintf('Saved %d trials to folder: %s\n', N, save_path);



%% 

% Load the data
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part1.mat');

% Compute length of each trial
trial_lengths = cellfun(@length, Filtered_200_Part2);

% Identify all unique lengths
unique_lengths = unique(trial_lengths);

% Find a group with at least 3 trials of the same length
valid_group_found = false;
for i = 1:length(unique_lengths)
    len = unique_lengths(i);
    idx = find(trial_lengths == len);
    if numel(idx) >= 100
        selected_indices = randsample(idx, 100);
        valid_group_found = true;
        break
    end
end

if ~valid_group_found
    error('No group of at least 3 trials with the same length found.');
end

% Extract and convert to a numeric matrix
selected_trials = Filtered_200_Part2(selected_indices);
Y = cell2mat(selected_trials); % size: N x 3
%Y = Y'; % transpose to get size: 3 x N

% Report
fprintf('Selected trial indices with same length: %d, %d, %d\n', selected_indices);
fprintf('Each trial has length: %d\n', len);
%%

















































%%

% Settings
freqs = 0:0.5:80;  % Higher resolution: 0.5 Hz bin width

% Function to generate synthetic frequency distribution
generate_distribution = @() create_custom_freq_distribution(freqs);

% Create 3 datasets
Y1 = generate_distribution();
Y2 = generate_distribution();
Y3 = generate_distribution();

% Plotting
figure;
subplot(3,1,1);
bar(freqs, Y1, 'FaceColor', [0.2 0.5 0.9], 'BarWidth', 1);
title('Frequency Distribution - Trial 1');
xlabel('Frequency (Hz)');
ylabel('Count');

subplot(3,1,2);
bar(freqs, Y2, 'FaceColor', [0.2 0.7 0.3], 'BarWidth', 1);
title('Frequency Distribution - Trial 2');
xlabel('Frequency (Hz)');
ylabel('Count');

subplot(3,1,3);
bar(freqs, Y3, 'FaceColor', [0.8 0.3 0.3], 'BarWidth', 1);
title('Frequency Distribution - Trial 3');
xlabel('Frequency (Hz)');
ylabel('Count');

% % Function Definition
% function Y = create_custom_freq_distribution(freqs)
%     Y = zeros(size(freqs));
% 
%     % Primary bump in 0–6 Hz
%     Y(freqs >= 0 & freqs <= 6) = 1000 + 200*randn(1, sum(freqs >= 0 & freqs <= 6));
% 
%     % Moderate flat 6–10 Hz
%     Y(freqs > 6 & freqs <= 10) = 300 + 50*randn(1, sum(freqs > 6 & freqs <= 10));
% 
%     % Secondary bump 10–20 Hz
%     Y(freqs > 10 & freqs <= 20) = 800 + 150*randn(1, sum(freqs > 10 & freqs <= 20));
% 
%     % Low 20–40 Hz
%     Y(freqs > 20 & freqs <= 40) = 100 + 30*randn(1, sum(freqs > 20 & freqs <= 40));
% 
%     % Very low 40–80 Hz
%     Y(freqs > 40) = 20 + 10*randn(1, sum(freqs > 40));
% 
%     % Sparse spikes in 70–80 Hz
%     spike_range = (freqs >= 70 & freqs <= 80);
%     spike_indices = find(spike_range);
%     spike_chance = rand(1, length(spike_indices)) > 0.8;
%     Y(spike_indices(spike_chance)) = Y(spike_indices(spike_chance)) + randi([100, 300], 1, sum(spike_chance));
% 
%     % Clip negatives
%     Y = max(Y, 0);
% end



%%


% Frequency bins: 0.5 Hz resolution
freqs = 0:1:80;

% Function to generate synthetic frequency distribution
generate_distribution = @() create_custom_freq_distribution(freqs);

% Generate 3 synthetic datasets
Y1 = generate_distribution();
Y2 = generate_distribution();
Y3 = generate_distribution();

% Plot Trial 1
figure;
bar(freqs, Y1, 'FaceColor', [0.2 0.5 0.9], 'BarWidth', 1);
title('Frequency Distribution - Trial 1');
xlabel('Frequency (Hz)');
ylabel('Count');

% Plot Trial 2
figure;
bar(freqs, Y2, 'FaceColor', [0.2 0.7 0.3], 'BarWidth', 1);
title('Frequency Distribution - Trial 2');
xlabel('Frequency (Hz)');
ylabel('Count');

% Plot Trial 3
figure;
bar(freqs, Y3, 'FaceColor', [0.8 0.3 0.3], 'BarWidth', 1);
title('Frequency Distribution - Trial 3');
xlabel('Frequency (Hz)');
ylabel('Count');

function Y = create_custom_freq_distribution(freqs)
    Y = zeros(size(freqs));

    % Primary bump: 0–10 Hz
    Y(freqs >= 0 & freqs <= 10) = 80 + 30 * randn(1, sum(freqs >= 0 & freqs <= 10));

    % Flat noisy region: 10–30 Hz
    Y(freqs > 10 & freqs <= 30) = 20 + 10 * randn(1, sum(freqs > 10 & freqs <= 30));

    % Very low Gaussian-like tail: 30–60 Hz
    range_30_60 = freqs > 30 & freqs <= 60;
    gauss_center = 45;
    gauss_width = 8;
    Y(range_30_60) = 10 * exp(-0.5 * ((freqs(range_30_60) - gauss_center) / gauss_width).^2) ...
                     + 2 * randn(1, sum(range_30_60));

    % Tiny bump in 60–80 Hz
    Y(freqs > 60 & freqs <= 80) = 5 + 2 * randn(1, sum(freqs > 60 & freqs <= 80));

    % Ensure non-negative
    Y = max(Y, 0);
end

