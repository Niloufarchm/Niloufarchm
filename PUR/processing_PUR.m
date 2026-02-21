% substitute the file for getting gain and position error data for all the
% groups in one table: repeat this for the PosError_LeftStruct and
% PosError_RightStruct to get the PE_aggregated
allDataCells = {};

for group = 1:6
    dir_name = sprintf('Group%d', group);

    right_data = load(fullfile(dir_name, 'rightDirection_avg_gain.mat'));
    left_data = load(fullfile(dir_name, 'leftDirection_avg_gain.mat'));

    if group <= 3 
        session = "S1";
    else 
        session = "S2";
    end
    
    if group == 1 || group == 4
        round = "R1";
    elseif group == 2 || group == 5
        round = "R2";
    else
        round = "R3";
    end

    subjects = fieldnames(right_data.rightDirection_avg_gain);
    for subj = 1:length(subjects)
        subject_name = subjects{subj};
        
        subjectData = table('Size', [1, 9], ...
                            'VariableTypes', ["string", "string", "string", "double", "double", "double", "double", "double", "double"], ...
                            'VariableNames', ["Subject", "Session", "Round", "GainRight5", "GainRight10", "GainRight20", "GainLeft5", "GainLeft10", "GainLeft20"]);
        subjectData.Subject = string(subject_name);
        subjectData.Session = session;
        subjectData.Round = round;
        
        for speed = 1:3
            if iscell(right_data.rightDirection_avg_gain.(subject_name))
                right_speed_value = right_data.rightDirection_avg_gain.(subject_name){speed};
            else
                right_speed_value = right_data.rightDirection_avg_gain.(subject_name)(speed);
            end
            
            if iscell(left_data.leftDirection_avg_gain.(subject_name))
                left_speed_value = left_data.leftDirection_avg_gain.(subject_name){speed};
            else
                left_speed_value = left_data.leftDirection_avg_gain.(subject_name)(speed);
            end
    
            if ~isnumeric(right_speed_value)
                right_speed_value = NaN; 
            end
            if ~isnumeric(left_speed_value)
                left_speed_value = NaN; 
            end
    
            subjectData{1, 3+speed} = right_speed_value;
            subjectData{1, 6+speed} = left_speed_value;
        end
        allDataCells{end+1} = subjectData;   
    end
end

Gain = vertcat(allDataCells{:});

save('Gain.mat', 'Gain');




% count of saccades for each direction
num_groups = 6;
speeds = [5, 10, 20];
columnNames = {'Subject', 'Session', 'Round'};

for speed = speeds
    columnNames = [columnNames, {sprintf('saccadeToLeft%d', speed), sprintf('saccadeToRight%d', speed)}];
end

saccadeDataRows = [];

for g = 1:num_groups
    session = sprintf('S%d', ceil(g / 3));  
    round = sprintf('R%d', mod(g - 1, 3) + 1);  

    group_folder = sprintf('Group%d', g);
    filenameLeft = fullfile(group_folder, 'saccadeFeaturesToLeftStruct.mat');
    filenameRight = fullfile(group_folder, 'saccadeFeaturesToRightStruct.mat');
    
    dataLeft = load(filenameLeft);
    dataRight = load(filenameRight);
    
    subjects = fieldnames(dataLeft.saccadeFeaturesToLeftStruct);
    
    for i = 1:numel(subjects)
        subject_name = subjects{i};
        newRow = {subject_name, session, round};
        
        for j = 1:numel(speeds)
            leftCount = 0;
            rightCount = 0;
            
            % Count for Left
            if isfield(dataLeft.saccadeFeaturesToLeftStruct, subject_name) && numel(dataLeft.saccadeFeaturesToLeftStruct.(subject_name)) >= j
                leftCount = numel(dataLeft.saccadeFeaturesToLeftStruct.(subject_name){j});
            end
            
            % Count for Right
            if isfield(dataRight.saccadeFeaturesToRightStruct, subject_name) && numel(dataRight.saccadeFeaturesToRightStruct.(subject_name)) >= j
                rightCount = numel(dataRight.saccadeFeaturesToRightStruct.(subject_name){j});
            end
            
            newRow = [newRow, {leftCount, rightCount}];
        end
        
        saccadeDataRows = [saccadeDataRows; newRow];
    end
end

Catchupsaccade_F = cell2table(saccadeDataRows, 'VariableNames', columnNames);

save('Catchupsaccade_F.mat', 'Catchupsaccade_F');

disp(Catchupsaccade_F(1:5,:));



% calculate velocity error using gain

VelocityErrorValues = zeros(size(Gain, 1), 6); 

gainColumns = {'GainRight5', 'GainRight10', 'GainRight20', 'GainLeft5', 'GainLeft10', 'GainLeft20'};

for i = 1:length(gainColumns)
    VelocityErrorValues(:, i) = abs(Gain{:, gainColumns{i}} - 1);
end

VelocityError = array2table(VelocityErrorValues, 'VariableNames', {'VERight5', 'VERight10', 'VERight20', 'VELeft5', 'VELeft10', 'VELeft20'});

VelocityError = [Gain(:, {'Subject', 'Session', 'Round'}), VelocityError];

save('VelocityError.mat', 'VelocityError');

disp(VelocityError(1:5, :));





% all the saccade features
num_groups = 6;
sessions = ["S1", "S2"];
rounds = ["R1", "R2", "R3"];
directions = ["ToLeft", "ToRight"];
speeds = [5, 10, 20];
saccadeFeatureNames = {'StartTime', 'EndTime', 'vPeak', 'Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Amplitude'};

variableNames = {'Subject', 'Session', 'Round', 'Direction', 'Speed', 'StartTime', 'EndTime', 'vPeak', 'Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Amplitude'};

saccadesCell = {};

for g = 1:num_groups
    session = sprintf('S%d', ceil(g / 3));
    round = sprintf('R%d', mod(g - 1, 3) + 1);
    
    filenameLeft = fullfile(sprintf('Group%d', g), 'saccadeFeaturesToLeftStruct.mat');
    filenameRight = fullfile(sprintf('Group%d', g), 'saccadeFeaturesToRightStruct.mat');
    dataLeft = load(filenameLeft);
    dataRight = load(filenameRight);
    
    subjects = fieldnames(dataLeft.saccadeFeaturesToLeftStruct);

    for subjIdx = 1:numel(subjects)
        subject_name = subjects{subjIdx};
        
        for directionIdx = 1:2
            if directionIdx == 1
                direction = "ToLeft";
                saccadeData = dataLeft.saccadeFeaturesToLeftStruct.(subject_name);
            else
                direction = "ToRight";
                saccadeData = dataRight.saccadeFeaturesToRightStruct.(subject_name);
            end

            for speedIdx = 1:numel(speeds)
                speed = speeds(speedIdx);
                saccadeDataForSpeed = saccadeData{speedIdx};
                if ~isempty(saccadeDataForSpeed)
                    for i = 1:numel(saccadeDataForSpeed)
                        saccade = saccadeDataForSpeed(i);
                        saccadesCell(end+1, :) = {subject_name, session, round, direction, speed, saccade.StartTime, saccade.EndTime, saccade.vPeak, saccade.Duration, saccade.StartX, saccade.StartY, saccade.EndX, saccade.EndY, saccade.Amplitude};
                    end
                end
            end
        end
    end
end

PUR_saccadesTable = cell2table(saccadesCell, 'VariableNames', variableNames);

save('allGroupsSaccadesTable.mat', 'PUR_saccadesTable');



% filtering the saccades for PUR


PUR_saccadesTable.av_vel = PUR_saccadesTable.Amplitude ./ (PUR_saccadesTable.Duration / 1000);
PUR_saccadesTable.ratio_av_Peak = PUR_saccadesTable.av_vel ./ PUR_saccadesTable.vPeak;

PUR_saccadesTable(PUR_saccadesTable.vPeak > 800, :) = [];
PUR_saccadesTable(PUR_saccadesTable.Amplitude > 14, :) = [];
PUR_saccadesTable(PUR_saccadesTable.Amplitude < 0.5, :) = [];
PUR_saccadesTable(PUR_saccadesTable.Duration > 60, :) = [];
PUR_saccadesTable(PUR_saccadesTable.Duration < 10, :) = [];

PUR_saccadesTable((PUR_saccadesTable.ratio_av_Peak < 0.5 | PUR_saccadesTable.ratio_av_Peak > 0.9), :) = [];


save('filtered_PUR_saccadesTable.mat', 'PUR_saccadesTable'); 





[G, groups] = findgroups(PUR_saccadesTable(:, {'Subject', 'Session', 'Round', 'Direction', 'Speed'}));

numberOfSaccades = splitapply(@length, PUR_saccadesTable.Subject, G);

summaryTable = [groups, table(numberOfSaccades)];

summaryTable.Properties.VariableNames(end) = {'NumberOfSaccades'};

% Assuming summaryTable is your current table

% Extract subject numbers
subjectNumbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), summaryTable.Subject);

% Create a numeric code for session and round combinations for custom ordering
% This assumes your sessions are named S1, S2 and your rounds are R1, R2, R3
sessionRoundCode = cellfun(@(s, r) str2double(s(2))*10 + str2double(r(2)), summaryTable.Session, summaryTable.Round);

% Add these as new columns to the table
summaryTable.SubjectNumbers = subjectNumbers;
summaryTable.SessionRoundCode = sessionRoundCode;

% Sort the table based on the new columns
% First by SubjectNumber, then by SessionRoundCode
sortedSummaryTable = sortrows(summaryTable, {'SubjectNumbers', 'SessionRoundCode'});

sortedSummaryTable.SubjectNumbers = [];
sortedSummaryTable.SessionRoundCode = [];




% Assuming sortedSummaryTable is your data table
directions = unique(sortedSummaryTable.Direction);
speeds = unique(sortedSummaryTable.Speed);

combNames = {};

for d = 1:length(directions)
    for s = 1:length(speeds)
        combName = sprintf('%s_%d', directions{d}, speeds(s));
        combNames{end+1} = combName;
    end
end

% Initialize a table for the results
uniqueGroups = unique(sortedSummaryTable(:, {'Subject', 'Session', 'Round'}), 'rows');
for i = 1:length(combNames)
    uniqueGroups.(combNames{i}) = zeros(height(uniqueGroups), 1);
end

for i = 1:height(sortedSummaryTable)
    row = sortedSummaryTable(i, :);
    combName = sprintf('%s_%d', row.Direction{1}, row.Speed);
    idx = find(all([strcmp(uniqueGroups.Subject, row.Subject), ...
                    strcmp(uniqueGroups.Session, row.Session), ...
                    strcmp(uniqueGroups.Round, row.Round)], 2));
    if ~isempty(idx)
        uniqueGroups.(combName)(idx) = row.NumberOfSaccades;
    end
end
subjectNumbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), uniqueGroups.Subject);

uniqueGroups.SubjectNumbers = subjectNumbers;

PUR_revised_data = sortrows(uniqueGroups, 'SubjectNumbers');

PUR_revised_data.SubjectNumbers = [];

disp(PUR_revised_data);

save('PUR_revised_data.mat', 'PUR_revised_data');





filename = 'PUR_revised_data.mat';

data = load(filename);
varName = fieldnames(data);
dataTable = data.(varName{1});


% Determine columns for mean calculation (from the 4th column onwards)
dataColumns = dataTable.Properties.VariableNames(4:end);

uniqueSubjects = unique(dataTable.Subject);

subjectNumbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), uniqueSubjects);
[~, sortIdx] = sort(subjectNumbers);
sortedSubjects = uniqueSubjects(sortIdx);

variableTypes = ['string', repmat({'double'}, 1, numel(dataColumns))];
PUR_revised_data_aggregated = table('Size', [0, numel(dataColumns) + 1], 'VariableTypes', variableTypes, 'VariableNames', ['Subject', dataColumns]);

for i = 1:numel(sortedSubjects)
    subj = sortedSubjects{i};
    subjData = dataTable(strcmp(dataTable.Subject, subj), :);
    
    meanValues = mean(table2array(subjData(:, dataColumns)), 1, 'omitnan');
    
    newRow = table(string(subj), 'VariableNames', {'Subject'});
    for j = 1:numel(dataColumns)
        newRow.(dataColumns{j}) = meanValues(j);
    end
    
    PUR_revised_data_aggregated = [PUR_revised_data_aggregated; newRow];
end

newFilename = ['aggregated' filename];
save(newFilename, 'PUR_revised_data_aggregated');







% common subjects
% Assuming the tables are loaded into the workspace from their .mat files
% and have consistent subject column names and formats

% Extract the subject lists from each table into variables
subjects_R = R_revised_data_aggregated.Subject;
subjects_PUR = PUR_revised_data_aggregated.Subject;
subjects_IQR = IQR_aggregated.Subject;
subjects_Latency = Latency_median_aggregated.Subject;
subjects_PE = PE_aggregated.Subject;
subjects_Gain = Gain_aggregated.Subject;
subjects_VelocityError = VelocityError_aggregated.Subject;
subjects_VergenceMean = VergenceMean_PUR_aggregated.Subject;
subjects_VergenceSD = VergenceSD_PUR_aggregated.Subject;
subjects_PE_MAX = PE_MAX_mean_aggregated.Subject;

% Find common subjects across all tables
commonSubjects = subjects_R; % Initialize with the first subject list
tablesSubjects = {subjects_PUR, subjects_IQR, subjects_Latency, subjects_PE, subjects_Gain, subjects_VelocityError, subjects_VergenceMean, subjects_VergenceSD, subjects_PE_MAX};

for i = 1:numel(tablesSubjects)
    commonSubjects = intersect(commonSubjects, tablesSubjects{i}, 'stable');
end

% Filter each table for common subjects
R_revised_data_aggregated = R_revised_data_aggregated(ismember(subjects_R, commonSubjects), :);
PUR_revised_data_aggregated = PUR_revised_data_aggregated(ismember(subjects_PUR, commonSubjects), :);
IQR_aggregated = IQR_aggregated(ismember(subjects_IQR, commonSubjects), :);
Latency_median_aggregated = Latency_median_aggregated(ismember(subjects_Latency, commonSubjects), :);
PE_aggregated = PE_aggregated(ismember(subjects_PE, commonSubjects), :);
Gain_aggregated = Gain_aggregated(ismember(subjects_Gain, commonSubjects), :);
VelocityError_aggregated = VelocityError_aggregated(ismember(subjects_VelocityError, commonSubjects), :);
VergenceMean_PUR_aggregated = VergenceMean_PUR_aggregated(ismember(subjects_VergenceMean, commonSubjects), :);
VergenceSD_PUR_aggregated = VergenceSD_PUR_aggregated(ismember(subjects_VergenceSD, commonSubjects), :);
PE_MAX_mean_aggregated = PE_MAX_mean_aggregated(ismember(subjects_PE_MAX, commonSubjects), :);

save('R_revised_data_aggregated.mat', 'R_revised_data_aggregated');
save('PUR_revised_data_aggregated.mat', 'PUR_revised_data_aggregated');
save('IQR_aggregated.mat', 'IQR_aggregated');
save('Latency_median_aggregated.mat', 'Latency_median_aggregated');
save('PE_aggregated.mat', 'PE_aggregated');
save('Gain_aggregated.mat', 'Gain_aggregated');
save('VelocityError_aggregated.mat', 'VelocityError_aggregated');
save('VergenceMean_PUR_aggregated.mat', 'VergenceMean_PUR_aggregated');
save('VergenceSD_PUR_aggregated.mat', 'VergenceSD_PUR_aggregated');
save('PE_MAX_mean_aggregated.mat', 'PE_MAX_mean_aggregated');

