folderPath = '';
targetFolderPath = '';
fileList = dir(fullfile(folderPath, '*.csv'));

if ~isempty(fileList)
    accumulateSaccadeData(folderPath, targetFolderPath, fileList);
else
    disp('No CSV files found in the specified folder.');
end





function accumulateSaccadeData(folderPath, targetFolderPath, fileList)
    
    PosError_LeftMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    PosError_RightMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    
    
    for i = 1:numel(fileList)
        fileName = fileList(i).name;
        subjectNumber = extractBetween(fileName, 'S_', '_S');
        subjectNumber = str2double(subjectNumber{1}(2:end)); % Remove the first digit
    
        data = readmatrix(fullfile(folderPath, fileName));
        
        Trials = struct(); 

        Trials.Samples.gx(1,:) = fillmissing(data(:,4), 'movmean', 100);
        Trials.Samples.gx(2,:) = fillmissing(data(:,6), 'movmean', 100);
        Trials.Samples.gy(1,:) = fillmissing(data(:,5), 'movmean', 100);
        Trials.Samples.gy(2,:) = fillmissing(data(:,7), 'movmean', 100);
        Trials.Samples.xT=fillmissing(data(:,8),'movmean',100);

        Trials.Samples.time = data(:,1);
        Trials.Header.rec.sample_rate = 250;
    
        Trials = edfExtractMicrosaccades_V3(Trials);
        
        speeds = [5, 10, 20]; % dva/s
        
        
        saccades = [Trials.Microsaccades.Start; Trials.Microsaccades.End]';


    
        % average of two eyes
        speeds = [5, 10, 20]; % dva/s
        targetSpeeds = [5, 10, 20];
        raw_xpos = Trials.Samples.gx(1,:);
        median_filtered_xpos = medfilt1(raw_xpos, 30); 
        xposL = movmean(fillmissing(median_filtered_xpos, 'movmean', 2100),200);
        
        raw_xpos = Trials.Samples.gx(2,:);
        median_filtered_xpos = medfilt1(raw_xpos, 30); 
        xposR = movmean(fillmissing(median_filtered_xpos, 'movmean', 2100),200); 
        
        xpos = (xposL + xposR) / 2;
        samplePeriod = 1 / 250;
        
        
        % target
        xposT = Trials.Samples.xT;
        
        tolerance = 0.01; 
        
        pause_points = find(...
            (Trials.Samples.xT >= -tolerance & Trials.Samples.xT <= tolerance) | ... % around 0
            (Trials.Samples.xT >= -15 - tolerance & Trials.Samples.xT <= -15 + tolerance) | ... % around -15
            (Trials.Samples.xT >= 15 - tolerance & Trials.Samples.xT <= 15 + tolerance)); % around 15
        
        % Find boundaries between pauses or start/end of a pause
        boundaries = find(diff(pause_points) > 1);
        
        % Sampling rate and delay in milliseconds
        sampling_rate = 250;
        delay_ms = 200;
        delay_samples = (sampling_rate / 1000) * delay_ms;
        
        pause_start = pause_points(1);
        pause_end = [];
        
        % Extract start and end indices of each pause
        for i = 1:length(boundaries)
            pause_end = [pause_end; pause_points(boundaries(i))];
            pause_start = [pause_start; pause_points(boundaries(i) + 1)];
        end
        pause_end = [pause_end; pause_points(end)];
        
        
        
        zeroCrossings = find(diff(sign(xposT)) ~= 0);
        
        ninth_zero_crossing = zeroCrossings(8);
        twenty_first_zero_crossing = zeroCrossings(20);
        
        changeIndex_10_dva_s = ninth_zero_crossing;
        changeIndex_20_dva_s = twenty_first_zero_crossing;

        
       
        
        
        
        % Number of x movement in each section
        num_xmovement_section1 = 8;
        num_xmovement_section2 = 12; 
        num_xmovement_section3 = 22; 
        
        
        movement_gains_left = cell(1, 3);  
        movement_gains_right = cell(1, 3);  
        
        pause_end = pause_end(pause_end > 0 & pause_end <= length(xposT)); 
        pause_start = pause_start(pause_start > 0 & pause_start <= length(xposT));
        
        for i = 1:length(pause_end) + 1
            % Determine the start index for each movement phase
            movement_start = 1;
            if i > 1
                movement_start = pause_end(i - 1) + 1;
            end
            
            % Check if movement_start is within the xposT range
            if movement_start > length(xposT)
                continue; % Skip if movement_start is out of range
            end
            
            % Determine the end index for each movement phase
            movement_end = length(xposT);
            if i <= length(pause_start)
                movement_end = pause_start(i) - 1;
            end
            
            % Check if movement_end is within the xposT range
            if movement_end < 1 || movement_end < movement_start
                continue; % Skip if movement_end is out of range or before movement_start
            end
            
            % To avoid noise, check the xposT value 100 indices after the pause
            check_index = min(movement_end + 100, length(xposT));
            
            % Determine the speed index based on the segment's end index
            if movement_end <= changeIndex_10_dva_s
                speed_index = 1;  % 5 dva/s
            elseif movement_end <= changeIndex_20_dva_s
                speed_index = 2;  % 10 dva/s
            else
                speed_index = 3;  % 20 dva/s
            end
            
            
        
             valid_positions = true(1, movement_end - movement_start + 1);
                
                for s = 1:length(Trials.Microsaccades.Start)
                    % Adjust saccade_start and saccade_end to be within the current segment
                    saccade_start = max(Trials.Microsaccades.Start(s), movement_start) - movement_start + 1;
                    saccade_end = min(Trials.Microsaccades.End(s), movement_end) - movement_start + 1;
                    if saccade_start <= saccade_end
                        valid_positions(saccade_start:saccade_end) = false;
                    end
                end
                
                % Calculate segment_gain_valid for the current segment
                segment_gain_valid = abs(xposT(movement_start:movement_end)) - abs(xpos(movement_start:movement_end));
                
                % Apply the valid_positions mask. Note: No need to adjust indices here as valid_positions now directly corresponds to the segment
                segment_gain_valid = segment_gain_valid(valid_positions);
                
                % Calculate the mean of valid segment gains
                mean_segment_gain = nanmean(segment_gain_valid);

        
                % Determine the direction of movement for assigning to the correct direction array
                if check_index <= length(xposT) && xposT(check_index) > xposT(movement_end)
                    % Movement to the right
                    movement_gains_right{speed_index} = [movement_gains_right{speed_index}, {mean_segment_gain}];
                elseif check_index <= length(xposT) && xposT(check_index) < xposT(movement_end)
                    % Movement to the left
                    movement_gains_left{speed_index} = [movement_gains_left{speed_index}, {mean_segment_gain}];
                end
             end         
      
       
        average_movement_gain_left = cell(1, 3);
        average_movement_gain_right = cell(1, 3);
        
        % Calculate for the left direction
        for speed_index = 1:3  
            % Get all sections for this speed
            sections = movement_gains_left{speed_index};
            medians = cellfun(@(x) median(x, 'omitnan'), sections, 'UniformOutput', false);
            % Filter out empty cells before calculating the mean of the medians
            medians = medians(~cellfun('isempty', medians));
            average_movement_gain_left{speed_index} = mean(cell2mat(medians));
        end
        
        % Calculate for the right direction
        for speed_index = 1:3  
            sections = movement_gains_right{speed_index};
            medians = cellfun(@(x) median(x, 'omitnan'), sections, 'UniformOutput', false);
            % Filter out empty cells before calculating the mean of the medians
            medians = medians(~cellfun('isempty', medians));
            average_movement_gain_right{speed_index} = mean(cell2mat(medians));
        end

        PosError_LeftMap(subjectNumber) = average_movement_gain_left;
        PosError_RightMap(subjectNumber) = average_movement_gain_right;
        
        
        movefile(fullfile(folderPath, fileName), targetFolderPath);
    end
    PosError_LeftStruct = map2struct(PosError_LeftMap);
    PosError_RightStruct = map2struct(PosError_RightMap);

    
    save(fullfile(targetFolderPath, 'PosError_LeftStruct.mat'), 'PosError_LeftStruct');
    save(fullfile(targetFolderPath, 'PosError_RightStruct.mat'), 'PosError_RightStruct');
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