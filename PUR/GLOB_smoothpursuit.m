folderPath = '';
targetFolderPath = '';
fileList = dir(fullfile(folderPath, '*.csv'));



if ~isempty(fileList)
    accumulateData(folderPath, targetFolderPath, fileList);
else
    disp('No CSV files found in the specified folder.');
end


function accumulateData(folderPath, targetFolderPath, fileList)    
    leftDirection_avg_gain = zeros(numel(fileList), 3);
    rightDirection_avg_gain = zeros(numel(fileList), 3);
    
    leftMatFile = fullfile(targetFolderPath, 'leftDirection_avg_gain.mat');
    rightMatFile = fullfile(targetFolderPath, 'rightDirection_avg_gain.mat');
    
    leftMatExists = exist(leftMatFile, 'file') == 2;
    rightMatExists = exist(rightMatFile, 'file') == 2;
    

    if leftMatExists
        load(leftMatFile, 'leftDirection_avg_gain');
    else
        left_avg_gain_matrix = [];
    end
    
    if rightMatExists
        load(rightMatFile, 'rightDirection_avg_gain');
    else
        right_avg_gain_matrix = [];
    end

   
   
     
    for i = 1:numel(fileList)
        fileName = fileList(i).name;

        subjectNumber = extractBetween(fileName, 'S_', '_S');
        subjectNumber = str2double(subjectNumber{1}(2:end)); % Remove the first digit
    
        
        data = readmatrix(fullfile(folderPath, fileName));
        Trials = struct(); 


        Trials.Samples.gx(1,:)=fillmissing(data(:,4),'movmean',100);
        Trials.Samples.gx(2,:)=fillmissing(data(:,6),'movmean',100);
        Trials.Samples.gy(1,:)=fillmissing(data(:,5),'movmean',100);
        Trials.Samples.gy(2,:)=fillmissing(data(:,7),'movmean',100);
        Trials.Samples.xT=fillmissing(data(:,8),'movmean',100);

        Trials.Samples.time=data(:,1);

        Trials.Header.rec.sample_rate=250;
    
        Trials=edfExtractMicrosaccades_V3(Trials)
        
        
        left_eye_velocity = Trials.Samples.velx(1,:);
        right_eye_velocity = Trials.Samples.velx(2,:);
        
        
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
        positionDifferences = diff(xpos);
        velocity = positionDifferences / samplePeriod;
        
        median_filtered_velocity = medfilt1(velocity, 10); 
        velxTrial = movmean(fillmissing(median_filtered_velocity, 'movmean', 1500),300); 
        
        
        % target
        xposT = Trials.Samples.xT;
        posdiffT = diff(xposT);
        Tvelocity = posdiffT / samplePeriod;
        velxT = movmean(Tvelocity, 100);
        
        
        
        zeropoints = find(velxTrial > -0.5 & velxTrial < 0.5);
        zeropointtrials = {};
        
        if ~isempty(zeropoints)
            current_cell = 1;
            zeropointtrials{current_cell} = [zeropoints(1)];
        
            for i = 2:length(zeropoints)
                if zeropoints(i) - zeropoints(i-1) > 200
                    current_cell = current_cell + 1;
                    zeropointtrials{current_cell} = [];
                end
                zeropointtrials{current_cell} = [zeropointtrials{current_cell}, zeropoints(i)];
            end
        end
        
        gains = cell(numel(targetSpeeds), 2); % 3 speeds x 2 directions (left and right)
        directionTitles = {'Left', 'Right'};
        speedTitles = {'5 dva/s', '10 dva/s', '20 dva/s'};
        
       % standard deviation multipliers for each speed section
        std_multiplier_5_dva_s = 1.2;
        std_multiplier_10_dva_s = 0.7;
        std_multiplier_20_dva_s = 0.4;
        
        % target velocity
        
        posdiffT = diff(xposT);
        samplePeriod = 1 / 250;  
        Tvelocity = posdiffT / samplePeriod;
        velxT = movmean(Tvelocity, 100);  % Smooth the target velocity
        
        % Find zero crossings in the target velocity
        zeroCrossings = find(diff(sign(velxT)) ~= 0);
        
        % Since the first zero crossing is the starting point, find the 9th and 21st zero crossings
        % Adjust the indices by adding 1 because the diff function reduces the length of the array by 1
        ninth_zero_crossing = zeroCrossings(17);
        twenty_first_zero_crossing = zeroCrossings(41);
        
        % Validate if we have enough zero crossings
        if length(zeroCrossings) < 21
            error('Not enough zero crossings detected to determine the speed sections.');
        end
        
        % These indices are the points right after the 8th and 20th oscillations end
        changeIndex_10_dva_s = ninth_zero_crossing;
        changeIndex_20_dva_s = twenty_first_zero_crossing;

        for i = 1:numel(zeropointtrials)-1
            trialStart = zeropointtrials{i}(end);
            if i < numel(zeropointtrials)
                trialEnd = zeropointtrials{i+1}(1) - 1;
            else
                trialEnd = length(velxTrial); 
            end
        
            % Determine which speed section we are in
            if trialEnd < changeIndex_10_dva_s
                std_multiplier = std_multiplier_5_dva_s;
                speedIndex = 1;  % Index for 5 dva/s
            elseif trialEnd < changeIndex_20_dva_s
                std_multiplier = std_multiplier_10_dva_s;
                speedIndex = 2;  % Index for 10 dva/s
            else
                std_multiplier = std_multiplier_20_dva_s;
                speedIndex = 3;  % Index for 20 dva/s
            end
        
            % Calculate standard deviation and apply threshold
            std_value = std(velxTrial(trialStart:trialEnd));
            threshold_velocity = max(abs(velxTrial(trialStart:trialEnd))) - (std_value * std_multiplier);
            indexes = find(abs(velxTrial(trialStart:trialEnd)) >= threshold_velocity);
        
            if ~isempty(indexes)
                sustained_velx_profile{i} = velxTrial(trialStart + indexes(1):trialStart + indexes(end));
                std_values(i) = std(sustained_velx_profile{i});
        
                meanEyeVelocity = mean(abs(sustained_velx_profile{i}));
        
                direction = sign(mean(velxT(trialStart:trialEnd)));
                directionIndex = direction < 0;  % 1 for left, 0 for right
        
                gain = meanEyeVelocity / speeds(speedIndex);
        
                gains{speedIndex, directionIndex + 1} = [gains{speedIndex, directionIndex + 1}, gain];
            else
                sustained_velx_profile{i} = [];
                std_values(i) = NaN;
            end
        end
        
        
        
        
        % mean gains
        meanGains = cell(size(gains)); 
        
        for speedIndex = 1:numel(targetSpeeds)
            for directionIndex = 1:2
                if ~isempty(gains{speedIndex, directionIndex})
                    meanGains{speedIndex, directionIndex} = mean(gains{speedIndex, directionIndex});
                else
                    meanGains{speedIndex, directionIndex} = NaN;  
                end
        
                
            end
        end
        
        for speedIndex = 1:3  
            % Left direction gains
            left_gains = gains{speedIndex, 1};  
            if ~isempty(left_gains)
                leftD_avg_gain_matrix(speedIndex) = mean(left_gains);
            end
            
            % Right direction gains
            right_gains = gains{speedIndex, 2};  
            if ~isempty(right_gains)
                rightD_avg_gain_matrix(speedIndex) = mean(right_gains);
            end
        end
        
        leftDirection_avg_gain(subjectNumber, :) = leftD_avg_gain_matrix;
        rightDirection_avg_gain(subjectNumber, :) = rightD_avg_gain_matrix;
        movefile(fullfile(folderPath, fileName), targetFolderPath);
    end
    leftMatFile = fullfile(targetFolderPath, 'leftDirection_avg_gain.mat');
    rightMatFile = fullfile(targetFolderPath, 'rightDirection_avg_gain.mat');

    save(leftMatFile, 'leftDirection_avg_gain');
    save(rightMatFile, 'rightDirection_avg_gain');
end

