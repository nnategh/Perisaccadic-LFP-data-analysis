fs = 1000;
num_samples = 200;
t = (0:num_samples-1) / fs;

frequencies = [5, 10, 15, 20, 25];
phases = [0, pi/6, pi/4, pi/3, pi/2];

signals = zeros(5, num_samples);
inst_phases = zeros(5, num_samples);

for i = 1:5
    signals(i,:) = sin(2*pi*frequencies(i)*t + phases(i));
    analytic_signal = hilbert(signals(i,:));
    inst_phases(i,:) = unwrap(angle(analytic_signal));
end

figure;
for i = 1:5
    subplot(5,2,2*i-1);
    plot(t, signals(i,:));
    title(sprintf('Time Series %d (f = %d Hz, phase = %.2f rad)', i, frequencies(i), phases(i)));
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    subplot(5,2,2*i);
    plot(t, inst_phases(i,:));
    title(sprintf('Instantaneous Phase %d', i));
    xlabel('Time (s)');
    ylabel('Phase (rad)');
end
saveas(gcf, 'Individual_Time_Series.png');

sum_signal = sum(signals, 1);
analytic_sum = hilbert(sum_signal);
inst_phase_sum = unwrap(angle(analytic_sum));

figure;
subplot(2,1,1);
plot(t, sum_signal);
title('Sum of Time Series');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t, inst_phase_sum);
title('Instantaneous Phase of Sum');
xlabel('Time (s)');
ylabel('Phase (rad)');
saveas(gcf, 'Sum_Time_Series.png');

for i = 1:5
    ts = signals(i,:);
    filename = sprintf('TimeSeries%d.mat', i);
    save(filename, 'ts');
end

ts_sum = sum_signal;
save('SumTimeSeries.mat', 'ts_sum');
%%

clear all;
clc; 
fs = 1000;
num_samples = 200;
t = (0:num_samples-1) / fs;

frequencies = [5, 10, 15, 20, 25];
phases = [0, pi/6, pi/4, pi/3, pi/2];

shuffled_signals = zeros(5, num_samples);
inst_phases = zeros(5, num_samples);

for i = 1:5
    signal = sin(2*pi*frequencies(i)*t + phases(i));
    bin_start = 50;
    bin_end = 150;
    bin_indices = bin_start:bin_end;
    selected_indices = randsample(bin_indices, 10);
    
    shuffled_signal = signal;
    values = shuffled_signal(selected_indices);
    shuffled_signal(selected_indices) = values(randperm(length(values)));
    
    shuffled_signals(i,:) = shuffled_signal;
    analytic_signal = hilbert(shuffled_signal);
    inst_phase = unwrap(angle(analytic_signal));
    inst_phases(i,:) = inst_phase;
    
    figure;
    subplot(2,1,1);
    plot(t, shuffled_signal);
    title(sprintf('Shuffled Time Series %d (f = %d Hz, phase = %.2f rad)', i, frequencies(i), phases(i)));
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    subplot(2,1,2);
    plot(t, inst_phase);
    title(sprintf('Instantaneous Phase %d', i));
    xlabel('Time (s)');
    ylabel('Phase (rad)');
    
    saveas(gcf, sprintf('Shuffled_TimeSeries_%d.png', i));
    close(gcf);
    
    ts = shuffled_signal;
    save(sprintf('Shuffled_TimeSeries_%d.mat', i), 'ts');
end

sum_signal = sum(shuffled_signals, 1);
analytic_sum = hilbert(sum_signal);
inst_phase_sum = unwrap(angle(analytic_sum));

figure;
subplot(2,1,1);
plot(t, sum_signal);
title('Sum of Shuffled Time Series');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t, inst_phase_sum);
title('Instantaneous Phase of Sum');
xlabel('Time (s)');
ylabel('Phase (rad)');

saveas(gcf, 'Shuffled_Sum_TimeSeries.png');
close(gcf);

ts_sum = sum_signal;
save('Shuffled_Sum_TimeSeries.mat', 'ts_sum');
%%
clear all; 
fs = 1000;
num_samples = 200;
t = (0:num_samples-1) / fs;

frequencies = [5, 10, 15, 20, 25];
phases = [0, pi/6, pi/4, pi/3, pi/2];

shuffled_signals = zeros(5, num_samples);
inst_phases = zeros(5, num_samples);

for i = 1:5
    signal = sin(2*pi*frequencies(i)*t + phases(i));
    
    % Define the bin (samples 50 to 150) and randomly select 20 samples to shuffle
    bin_start = 50;
    bin_end = 150;
    bin_indices = bin_start:bin_end;
    selected_indices = randsample(bin_indices, 20);
    
    shuffled_signal = signal;
    values = shuffled_signal(selected_indices);
    shuffled_signal(selected_indices) = values(randperm(length(values)));
    
    shuffled_signals(i,:) = shuffled_signal;
    analytic_signal = hilbert(shuffled_signal);
    inst_phase = unwrap(angle(analytic_signal));
    inst_phases(i,:) = inst_phase;
    
    figure;
    subplot(2,1,1);
    plot(t, shuffled_signal);
    title(sprintf('Shuffled Time Series %d (f = %d Hz, phase = %.2f rad)', i, frequencies(i), phases(i)));
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    subplot(2,1,2);
    plot(t, inst_phase);
    title(sprintf('Instantaneous Phase %d', i));
    xlabel('Time (s)');
    ylabel('Phase (rad)');
    
    saveas(gcf, sprintf('Shuffled_TimeSeries_%d.png', i));
    close(gcf);
    
    ts = shuffled_signal;
    save(sprintf('Shuffled_TimeSeries_%d.mat', i), 'ts');
end

sum_signal = sum(shuffled_signals, 1);
analytic_sum = hilbert(sum_signal);
inst_phase_sum = unwrap(angle(analytic_sum));

figure;
subplot(2,1,1);
plot(t, sum_signal);
title('Sum of Shuffled Time Series');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t, inst_phase_sum);
title('Instantaneous Phase of Sum');
xlabel('Time (s)');
ylabel('Phase (rad)');

saveas(gcf, 'Shuffled_Sum_TimeSeries.png');
close(gcf);

ts_sum = sum_signal;
save('Shuffled_Sum_TimeSeries.mat', 'ts_sum');
%%

clear all; 
fs = 1000;
num_samples = 200;
t = (0:num_samples-1) / fs;

frequencies = [5, 10, 15, 20, 25];
phases = [0, pi/6, pi/4, pi/3, pi/2];

signals = zeros(5, num_samples);
inst_phases = zeros(5, num_samples);

for i = 1:5
    signals(i,:) = sin(2*pi*frequencies(i)*t + phases(i));
    analytic_signal = hilbert(signals(i,:));
    inst_phases(i,:) = unwrap(angle(analytic_signal));
end

figure;
for i = 1:5
    subplot(5,2,2*i-1);
    plot(t, signals(i,:));
    title(sprintf('Time Series %d (f = %d Hz, phase = %.2f rad)', i, frequencies(i), phases(i)));
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    subplot(5,2,2*i);
    plot(t, inst_phases(i,:));
    title(sprintf('Instantaneous Phase %d', i));
    xlabel('Time (s)');
    ylabel('Phase (rad)');
end
saveas(gcf, 'Individual_Time_Series.png');

sum_signal = sum(signals, 1);

% Shuffle only the sum of time series
shuffled_sum_signal = sum_signal;
bin_start = 50;
bin_end = 150;
bin_indices = bin_start:bin_end;
selected_indices = randsample(bin_indices, 20);
temp_values = shuffled_sum_signal(selected_indices);
shuffled_sum_signal(selected_indices) = temp_values(randperm(length(temp_values)));

analytic_sum = hilbert(shuffled_sum_signal);
inst_phase_sum = unwrap(angle(analytic_sum));

figure;
subplot(2,1,1);
plot(t, shuffled_sum_signal);
title('Shuffled Sum of Time Series');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t, inst_phase_sum);
title('Instantaneous Phase of Shuffled Sum');
xlabel('Time (s)');
ylabel('Phase (rad)');
saveas(gcf, 'Shuffled_Sum_Time_Series.png');

for i = 1:5
    ts = signals(i,:);
    filename = sprintf('TimeSeries%d.mat', i);
    save(filename, 'ts');
end

ts_sum = shuffled_sum_signal;
save('Shuffled_SumTimeSeries.mat', 'ts_sum');

