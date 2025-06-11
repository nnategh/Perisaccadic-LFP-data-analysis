clear; close all; clc;

% Load necessary data
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Clean_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/saccade_onset.mat')
%load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Valid_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/raise_times.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_50_Corrected.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_100_Corrected.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_200_Corrected.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_400_Corrected.mat')
%%
fs = 1000;                       
nSamples = size(Cue1LFP, 2);

filtered_data_sets = {
    Cue1LFP_Filtered_50, '50';
    Cue1LFP_Filtered_100, '100';
    Cue1LFP_Filtered_200, '200';
    Cue1LFP_Filtered_400, '400';
};

for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    bandName = filtered_data_sets{f, 2};
    
    part1Cell = cell(numel(Clean_Trials), 1);
    part2Cell = cell(numel(Clean_Trials), 1);
    part3Cell = cell(numel(Clean_Trials), 1);
    part4Cell = cell(numel(Clean_Trials), 1);
    
    for i = 1:numel(Clean_Trials)
        trialNum = Clean_Trials(i);
        
        saccade_time = double(saccade_onset(trialNum)) / fs;
        raise_time   = double(raise_times(trialNum)) / fs;
        
        start_time   = 0;
        split1_time  = max(0, saccade_time - 0.1);
        split2_time  = saccade_time + 0.05;
        split3_time  = raise_time - 0.05;
        end_time     = nSamples / fs;
        
        idx_start  = 1;
        idx_split1 = max(1, round(split1_time * fs));
        idx_split2 = max(1, round(split2_time * fs));
        idx_split3 = max(1, round(split3_time * fs));
        idx_end    = nSamples;
        
        part1 = filtered_data(trialNum, idx_start:idx_split1);
        part2 = filtered_data(trialNum, idx_split1:idx_split2);
        part3 = filtered_data(trialNum, idx_split2:idx_split3);
        part4 = filtered_data(trialNum, idx_split3:idx_end);
        
        part1Cell{i} = part1;
        part2Cell{i} = part2;
        part3Cell{i} = part3;
        part4Cell{i} = part4;
    end
    
    varName1 = sprintf('Filtered_%s_Part1', bandName);
    varName2 = sprintf('Filtered_%s_Part2', bandName);
    varName3 = sprintf('Filtered_%s_Part3', bandName);
    varName4 = sprintf('Filtered_%s_Part4', bandName);
    
    eval([varName1 ' = part1Cell;']);
    eval([varName2 ' = part2Cell;']);
    eval([varName3 ' = part3Cell;']);
    eval([varName4 ' = part4Cell;']);
    
    save(sprintf('%s.mat', varName1), varName1);
    save(sprintf('%s.mat', varName2), varName2);
    save(sprintf('%s.mat', varName3), varName3);
    save(sprintf('%s.mat', varName4), varName4);
end
%%
clc; 
clear all;

clc;
clear;

% Manually load each .mat file
data10 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_10.mat');
data20 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_20.mat');
data30 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_30.mat');
data40 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_40.mat');

% Retrieve the variable from each loaded structure automatically.
% This avoids hardcoding the field name, which may not match what is saved.
var10 = data10.(fieldnames(data10){1});
var20 = data20.(fieldnames(data20){1});
var30 = data30.(fieldnames(data30){1});
var40 = data40.(fieldnames(data40){1});

% Merge the frequency results from each file into one variable.
mergedFreq_F_200_Part_1 = [var10; var20; var30; var40];

% Save the merged frequency results into a single .mat file.
save('Filtered_200_Part1_Frequency.mat', 'mergedFreq_F_200_Part_1');

%%

clc;
clear;

% Manually load each .mat file
data10 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_10.mat');
data20 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_20.mat');
data30 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_30.mat');
data40 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/F200_Part1/Filtered_200_Part1_F_40.mat');

% Retrieve the variable from each loaded structure by first obtaining the field names.
f10 = fieldnames(data10);
var10 = data10.(f10{1});

f20 = fieldnames(data20);
var20 = data20.(f20{1});

f30 = fieldnames(data30);
var30 = data30.(f30{1});

f40 = fieldnames(data40);
var40 = data40.(f40{1});

% Merge the frequency results from each file into one variable.
mergedFreq_F_200_Part_1 = [var10; var20; var30; var40];

% Save the merged frequency results into a single .mat file.
save('Filtered_200_Part1_Frequency.mat', 'mergedFreq_F_200_Part_1');


%%
clear; clc;

% Load the merged frequency data (cell array)
load('Filtered_200_Part1_Frequency.mat', 'mergedFreq_F_200_Part_1');

N = numel(mergedFreq_F_200_Part_1);  % number of trials
numSquares = 5;  % maximum number of components per trial

% Gather all available component values (from all trials) to determine the colormap limits
allVals = [];
for i = 1:N
    comp = mergedFreq_F_200_Part_1{i};
    allVals = [allVals; comp(:)];  %#ok<AGROW>
end

minVal = min(allVals);
maxVal = max(allVals);

% Choose a colormap (here, using jet with 256 colors)
cmap = jet(256);
nColors = size(cmap,1);

figure; hold on;

% Loop over each trial
% We'll use a coordinate system where each trial occupies one row.
% For each trial i, we plot 5 squares along the x-axis.
% The bottom-left of the square for component j in trial i is at: [j-1, i-1].
for i = 1:N
    comp = mergedFreq_F_200_Part_1{i};
    nComp = length(comp);
    for j = 1:numSquares
        % Determine rectangle position:
        xPos = j - 1;      % x from 0 to 4
        yPos = i - 1;      % y from 0 to N-1
        width = 1;
        height = 1;
        
        if j <= nComp
            % Map the component value to a color using linear scaling.
            normVal = (comp(j) - minVal) / (maxVal - minVal);
            colorIdx = round(normVal*(nColors-1)) + 1;
            rectColor = cmap(colorIdx, :);
        else
            % If no component exists for this square, use black.
            rectColor = [0 0 0];
        end
        
        rectangle('Position', [xPos, yPos, width, height], ...
                  'FaceColor', rectColor, 'EdgeColor', 'w');
    end
end

% Adjust the axes.
xlim([0, numSquares]);
ylim([0, N]);
set(gca, 'YDir', 'normal');  % so trial 1 is at the bottom
set(gca, 'XTick', 0.5:1:(numSquares-0.5), 'XTickLabel', 1:numSquares);
set(gca, 'YTick', 0.5:1:(N-0.5), 'YTickLabel', 1:N);
xlabel('Component Index');
ylabel('Trial Number');
title('Filtered 200 Part1 Frequency Components');

% Add a colorbar with the appropriate colormap and value range.
colormap(cmap);
caxis([minVal maxVal]);
colorbar;
% Save the plot as PNG and FIG files.
saveas(gcf, 'Filtered_200_Part1_Frequency.png');
savefig('Filtered_200_Part1_Frequency.fig');