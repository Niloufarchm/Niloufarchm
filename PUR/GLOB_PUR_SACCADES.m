folderPath = '';
targetFolderPath = '';
fileList = dir(fullfile(folderPath, '*.csv'));

if ~isempty(fileList)
    accumulateSaccadeData(folderPath, targetFolderPath, fileList);
else
    disp('No CSV files found in the specified folder.');
end


function accumulateSaccadeData(folderPath, targetFolderPath, fileList)
        saccadeFeaturesToLeftMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
        saccadeFeaturesToRightMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

    for i = 1:numel(fileList)
        fileName = fileList(i).name;
        subjectNumber = extractBetween(fileName, 'S_', '_S');
        subjectNumber = str2double(subjectNumber{1}(2:end)); % Remove the first digit
    
        data = readmatrix(fullfile(folderPath, fileName));
        
        Trials = struct(); 

        leftEyeProcessed = fillmissing(data(:,4), 'movmean', 100);
        rightEyeProcessed = fillmissing(data(:,6), 'movmean', 100);
        
        Trials.Samples.gx(1,:) = (leftEyeProcessed + rightEyeProcessed) / 2;
        Trials.Samples.gx(2,:) = (leftEyeProcessed + rightEyeProcessed) / 2;

        
        leftEyeVerticalProcessed = fillmissing(data(:,5), 'movmean', 100);
        rightEyeVerticalProcessed = fillmissing(data(:,7), 'movmean', 100);
        
        Trials.Samples.gy(1,:) = (leftEyeVerticalProcessed + rightEyeVerticalProcessed) / 2;
        Trials.Samples.gy(2,:) = (leftEyeVerticalProcessed + rightEyeVerticalProcessed) / 2;



        Trials.Samples.time=data(:,1);
        Trials.Header.rec.sample_rate=250;
    
        Trials=edfExtractMicrosaccades_V3(Trials)
        
        
        left_eye_velocity = Trials.Samples.velx(1,:);
        right_eye_velocity = Trials.Samples.velx(2,:);
        
        
        speeds = [5, 10, 20]; % dva/s
        
        
        raw_velocity = Trials.Samples.normvelx(1,:);
        median_filtered_velocity = medfilt1(raw_velocity, 50); 
        interpolated_velocity = fillmissing(median_filtered_velocity, 'movmean', 200); 
        velxTrialL = movmean(interpolated_velocity, 200);         
        zeropointsL = find(velxTrialL > -0.1 & velxTrialL < 0.1);
        zeropointtrialsL = {};
        
        if ~isempty(zeropointsL)
            current_cell = 1;
            zeropointtrialsL{current_cell} = [zeropointsL(1)];
        
            for j = 2:length(zeropointsL)
                if zeropointsL(j) - zeropointsL(j-1) > 100
                    current_cell = current_cell + 1;
                    zeropointtrialsL{current_cell} = [];
                end
                zeropointtrialsL{current_cell} = [zeropointtrialsL{current_cell}, zeropointsL(j)];
            end
        end
        sustained_velx_profileL = cell(1, numel(zeropointtrialsL)-1);
        std_values = zeros(1, numel(zeropointtrialsL)-1);
        
        for i = 1:numel(zeropointtrialsL)-1
            trialStartL = zeropointtrialsL{i}(end);
            trialEndL = zeropointtrialsL{i+1}(1);
            
            [~, max_index] = max(abs(velxTrialL(trialStartL:trialEndL)));
            
            max_velocity = abs(velxTrialL(trialStartL+max_index));
            threshold_velocity = max_velocity - 1;
            
            % sustained phase indexes around the highest index
            std_value = std(abs(velxTrialL(trialStartL:trialEndL)));
            indexes = find(abs(velxTrialL(trialStartL:trialEndL)) >= threshold_velocity - (std_value)/4);
            
            % velocity profile for the sustained phase
            sustained_velx_profileL{i} = abs(velxTrialL(trialStartL+indexes(1):trialStartL+indexes(end)));
            
            % standard deviation for the sustained phase
            std_values(i) = std(sustained_velx_profileL{i});
        end
        trialsL = cell(1, 3);
        
        % Loop through sustained_velx_profileL cells from the first cell to the 10th cell
        for j = 1:min(10, numel(sustained_velx_profileL))
            max_velocity = max(sustained_velx_profileL{j});
            if max_velocity >= 4 && max_velocity <= 9
                trialsL{1}{end+1} = sustained_velx_profileL{j};
            end
        end
        
        % Loop through sustained_velx_profileL cells from the 8th to the 20th cell
        for j = 8:min(20, numel(sustained_velx_profileL))
            max_velocity = max(sustained_velx_profileL{j});
            if max_velocity >= 6 && max_velocity <= 15
                trialsL{2}{end+1} = sustained_velx_profileL{j};
            end
        end
        
        % Loop through sustained_velx_profileL cells from the 18th cell to the last cell
        for j = 19:numel(sustained_velx_profileL)
            max_velocity = max(sustained_velx_profileL{j});
            if max_velocity > 8
                trialsL{3}{end+1} = sustained_velx_profileL{j};
            end
        end
        
        avg_gainL = zeros(1, numel(trialsL));
        
        for j = 1:numel(trialsL)
            trial_dataL = trialsL{j};
            trial_speed = speeds(j);
            
            avg_gainL(j) = nanmean(double(trial_dataL{1}) / trial_speed);
        end
        
        trialIndexes = cell(1, 3);
        
        for j = 1:numel(trialsL)
            current_trial = trialsL{j};
            current_trial_indexes = cell(1, numel(current_trial));
            
            for k = 1:numel(current_trial)
                current_range = current_trial{k};
                [~, start_index] = ismember(current_range(1), abs(velxTrialL));
                [~, end_index] = ismember(current_range(end), abs(velxTrialL));
                
                current_trial_indexes{k} = start_index:end_index;
            end
            
            trialIndexes{j} = current_trial_indexes;
        end
        
        left_velocitiesL = cell(1, numel(trialsL));
        right_velocitiesL = cell(1, numel(trialsL));
        
        for j = 1:numel(trialsL)
            current_trial = trialsL{j};
            current_indexes = trialIndexes{j};
        
            left_velocities = {};
            right_velocities = {};
        
            for k = 1:numel(current_trial)
                current_range = current_trial{k};
                current_range_indexes = current_indexes{k};
        
                if isempty(current_range_indexes)
                    continue;
                end
        
                % Check the position of the first and second indexes
                start_index = current_range_indexes(1);
                end_index = current_range_indexes(end);
                start_position = Trials.Samples.gx(1, start_index);
                end_position = Trials.Samples.gx(1, end_index+1);
        
                if end_position > start_position
                    right_velocities{end+1} = current_range;
                else
                    left_velocities{end+1} = current_range;
                end
            end
        
            left_velocitiesL{j} = left_velocities;
            right_velocitiesL{j} = right_velocities;
        end
        


        % Determine leftTrialIndexes and rightTrialIndexes
        leftTrialIndexes = getTrialIndexes(Trials, trialsL, velxTrialL, 'left');
        rightTrialIndexes = getTrialIndexes(Trials, trialsL, velxTrialL, 'right');

        saccadeFeaturesToLeft = cell(1, numel(leftTrialIndexes));
        saccadeFeaturesToRight = cell(1, numel(rightTrialIndexes));

        for j = 1:numel(leftTrialIndexes)
            saccadeFeaturesToLeft{j} = getSaccadesForTrial(leftTrialIndexes{j}, Trials);
        end

        for j = 1:numel(rightTrialIndexes)
            saccadeFeaturesToRight{j} = getSaccadesForTrial(rightTrialIndexes{j}, Trials);
        end

        saccadeFeaturesToLeftMap(subjectNumber) = saccadeFeaturesToLeft;
        saccadeFeaturesToRightMap(subjectNumber) = saccadeFeaturesToRight;
        movefile(fullfile(folderPath, fileName), targetFolderPath);
    end
    saccadeFeaturesToLeftStruct = map2struct(saccadeFeaturesToLeftMap);
    saccadeFeaturesToRightStruct = map2struct(saccadeFeaturesToRightMap);

    save(fullfile(targetFolderPath, 'saccadeFeaturesToLeftStruct.mat'), 'saccadeFeaturesToLeftStruct');
    save(fullfile(targetFolderPath, 'saccadeFeaturesToRightStruct.mat'), 'saccadeFeaturesToRightStruct');
end

function mapStruct = map2struct(mapObj)
    keys = mapObj.keys();
    values = mapObj.values();
    mapStruct = struct;
    for i = 1:length(keys)
        keyName = sprintf('subject_%d', keys{i});
        mapStruct.(keyName) = values{i};
    end
end   
    

function trialIndexes = getTrialIndexes(Trials, trials, velxTrial, direction)
    trialIndexes = cell(1, numel(trials));
    for j = 1:numel(trials)
        current_trial = trials{j};
        current_trial_indexes = cell(1, numel(current_trial));
    
        for k = 1:numel(current_trial)
            current_range = current_trial{k};
            [~, start_index] = ismember(current_range(1), abs(velxTrial));
            [~, end_index] = ismember(current_range(end), abs(velxTrial));
        
            if start_index == 0 || end_index == 0
                current_trial_indexes{k} = [];
            else
                current_trial_indexes{k} = start_index:end_index;
            end
        end

        direction_indexes = {};

        for k = 1:numel(current_trial)
            current_range = current_trial{k};
            current_range_indexes = current_trial_indexes{k};

            if isempty(current_range_indexes)
                continue;
            end

            start_index = current_range_indexes(1);
            end_index = current_range_indexes(end);
            start_position = Trials.Samples.gx(2, start_index);
            end_position = Trials.Samples.gx(2, end_index+1);

            if strcmp(direction, 'left') && end_position <= start_position
                direction_indexes{end+1} = current_range_indexes;
            elseif strcmp(direction, 'right') && end_position > start_position
                direction_indexes{end+1} = current_range_indexes;
            end
        end

        trialIndexes{j} = direction_indexes;
    end
    return
end

function features = getSaccadesForTrial(trialIndexes, Trials)
    MicrosaccadesStart = Trials.Microsaccades.Start;  
    MicrosaccadesEnd = Trials.Microsaccades.End;
    MicrosaccadesvPeak = Trials.Microsaccades.vPeak;
    MicrosaccadesDuration = Trials.Microsaccades.Duration;
    MicrosaccadesStartX = Trials.Microsaccades.StartX;
    MicrosaccadesStartY = Trials.Microsaccades.StartY;
    MicrosaccadesEndX = Trials.Microsaccades.EndX;
    MicrosaccadesEndY = Trials.Microsaccades.EndY;
    MicrosaccadesAmplitude = Trials.Microsaccades.Amplitude;

    trial_features = [];
    
    for k = 1:numel(trialIndexes)
        current_trial_indices = trialIndexes{k};
        if ~isempty(current_trial_indices) && iscell(current_trial_indices)
            current_trial_indices = current_trial_indices{1};
        end
            
        for j = 1:numel(MicrosaccadesStart)
            if MicrosaccadesStart(j) >= min(current_trial_indices) && MicrosaccadesStart(j) <= max(current_trial_indices)
                saccade = struct();
                saccade.StartTime = MicrosaccadesStart(j);
                saccade.EndTime = MicrosaccadesEnd(j);
                saccade.vPeak = MicrosaccadesvPeak(j);
                saccade.Duration = MicrosaccadesDuration(j);
                saccade.StartX = MicrosaccadesStartX(j);
                saccade.StartY = MicrosaccadesStartY(j);
                saccade.EndX = MicrosaccadesEndX(j);
                saccade.EndY = MicrosaccadesEndY(j);
                saccade.Amplitude = MicrosaccadesAmplitude(j);


                trial_features = [trial_features, saccade];
            end
        end
    end
    
    features = trial_features;
end


