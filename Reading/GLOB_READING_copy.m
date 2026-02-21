
% SACCADE FEATURES
% MICROSACCADE SPATIAL ENTROPY method changed
% OTHERMETRICS


folderPath = '';
targetFolderPath = '';
fileList = dir(fullfile(folderPath, '*.csv'));


if ~isempty(fileList)
    accumulateData(folderPath, targetFolderPath, fileList);
else
    disp('No CSV files found in the specified folder.');
end

function accumulateData(folderPath, targetFolderPath, fileList)
    saccadefeaturesMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    OtherMetricsMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    SpatialEntropyMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

    
    
    for i = 1:numel(fileList)
        fileName = fileList(i).name;

        subjectNumber = extractBetween(fileName, 'S_', '_S');
        subjectNumber = str2double(subjectNumber{1}(2:end)); % Remove the first digit
    
        
        data = readmatrix(fullfile(folderPath, fileName));
        Trials = struct(); 

        display.dist=60;
        display.width=56.58;
        display.resolution=[1280,720];

        
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


        
        % 1) time that takes for the whole task in second

        Duration = (Trials.Samples.time(end))/1000;
        
        % 2) number of saccades per second
        amplitudes = Trials.Microsaccades.Amplitude;

        microsaccadeIndices = [];
        saccadeIndices = [];
        
        for i = 1:length(amplitudes)
            if amplitudes(i) < 1
                %  microsaccade
                microsaccadeIndices = [microsaccadeIndices, i];
            elseif amplitudes(i) >= 1 && amplitudes(i) <= 15
                %  saccade
                saccadeIndices = [saccadeIndices, i];
            end
        end
        
        microsaccadesStartTimes = Trials.Microsaccades.StartTime(microsaccadeIndices);
        microsaccadesEndTimes = Trials.Microsaccades.EndTime(microsaccadeIndices);
        
        saccadesStartTimes = Trials.Microsaccades.StartTime(saccadeIndices);
        saccadesEndTimes = Trials.Microsaccades.EndTime(saccadeIndices);

        
        totalSaccades = length(saccadesStartTimes);
        saccadesPerSecond = totalSaccades / Duration;

        totalMicroSaccades = length(microsaccadesStartTimes);
        microsaccadePerSecond = totalMicroSaccades/ Duration;


        




        % 3) number of fixation per duration
         
        
        % Number of fixations is typically one more than the number of saccades
        numFixations = totalSaccades + 1;
        
        % Fixations per duration
        numfixationsOnDuration = numFixations / Duration;
        
        % 4 & 5) each fixation duration in ms and mean and sd
        
        fixationDurations = zeros(totalSaccades + 1, 1);
        
        % First fixation duration
        if totalSaccades > 0
            fixationDurations(1) = saccadesStartTimes(1) - Trials.Samples.time(1);
        else
            % In case there are no saccades, the whole trial is one long fixation
            fixationDurations(1) = Trials.Samples.time(end) - Trials.Samples.time(1);
        end
        
        % Intermediate fixation durations
        for j = 1:totalSaccades-1
            fixationDurations(j+1) = saccadesStartTimes(j+1) - saccadesEndTimes(j);
        end
        
        % Last fixation duration
        if totalSaccades > 0
            fixationDurations(end) = Trials.Samples.time(end) - saccadesEndTimes(end);
        end
        
        % fixationDurations are in milliseconds
        
        % mean and sd for fixation durations
        meanFixationDuration = mean(fixationDurations);
        sdFixationDuration = std(fixationDurations);
       


        % 6 & 7) number of regression, and rate: on saccade
        
        backwardSaccades = find(Trials.Microsaccades.DeltaX(saccadeIndices) < 0);
        regressionIndices = saccadeIndices(backwardSaccades);
        regressionAmplitudes = amplitudes(regressionIndices);
        numRegressions = sum(regressionAmplitudes < 15);
        
        % rate of regressions per saccade and second
        if totalSaccades > 0
            regressionsOnSaccade = numRegressions / totalSaccades;
            regressionPerSecond = numRegressions /Duration;
        else
            regressionsOnSaccade = 0;
            regressionPerSecond = 0;
        end

        progressiveSaccades = find(Trials.Microsaccades.DeltaX(saccadeIndices) > 0);
        progressiveIndices = saccadeIndices(progressiveSaccades);
        progressiveAmplitudes = amplitudes(progressiveIndices);
        numProgressions = sum( progressiveAmplitudes < 15);
        
        % rate of progressions per saccade and second
        if totalSaccades > 0
            ProgressionPerSecond = numProgressions /Duration;
        else
            ProgressionPerSecond = 0;
        end

        microsaccadesStartTimes = Trials.Microsaccades.StartTime(microsaccadeIndices);
        microsaccadesEndTimes = Trials.Microsaccades.EndTime(microsaccadeIndices);
        
        saccadesStartTimes = Trials.Microsaccades.StartTime(saccadeIndices);
        saccadesEndTimes = Trials.Microsaccades.EndTime(saccadeIndices);

        
        microsaccadesStartX = Trials.Microsaccades.StartX(microsaccadeIndices);
        microsaccadesEndX = Trials.Microsaccades.EndX(microsaccadeIndices);
        
        microsaccadesStartY = Trials.Microsaccades.StartY(microsaccadeIndices);
        microsaccadesEndY = Trials.Microsaccades.EndY(microsaccadeIndices);
        
        relativeEndX = microsaccadesEndX - microsaccadesStartX;
        relativeEndY = microsaccadesEndY - microsaccadesStartY;


        % we changed the method of calculating spatial entropy: you can the updated code for spatial entropy calcultaion at the end of the Reading_Final.m file.                
%         numBins = 50; 
%         h = histogram2(relativeEndX, relativeEndY, numBins, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on', 'Normalization', 'probability', 'Visible', 'off');
%         img = h.Values;
%         entropyVal = entropy(img);
%         
%         SpatialEntropyMap(subjectNumber) = entropyVal;
%         
%         
%         SpatialEntropyMap(subjectNumber) = struct('RelativeEndX', relativeEndX, 'RelativeEndY', relativeEndY, 'Entropy', entropyVal);
%         
        saccadefeatures = getReadingSaccades(Trials);
        saccadefeaturesMap(subjectNumber) = saccadefeatures;

        metrics = struct();
        metrics.Duration = Duration;
        metrics.saccadesPerSecond = saccadesPerSecond;
        metrics.numfixationsOnDuration = numfixationsOnDuration;
        metrics.meanFixationDuration = meanFixationDuration;
        metrics.sdFixationDuration = sdFixationDuration;
        metrics.numRegressions = numRegressions;
        metrics.regressionsOnSaccade = regressionsOnSaccade;
        metrics.microsaccadePerSecond = microsaccadePerSecond;
        metrics.regressionPerSecond = regressionPerSecond;
        metrics.ProgressionPerSecond = ProgressionPerSecond;


        OtherMetricsMap(subjectNumber) = metrics;
        movefile(fullfile(folderPath, fileName), targetFolderPath);      
    end
    saccadefeaturesStruct = map2struct(saccadefeaturesMap);
    OtherMetricsStruct = map2struct(OtherMetricsMap);
    
%     save(fullfile(targetFolderPath, 'SpatialEntropyStruct.mat'), 'SpatialEntropyStruct');
    save(fullfile(targetFolderPath, 'saccadefeaturesStruct.mat'), 'saccadefeaturesStruct');
    save(fullfile(targetFolderPath, 'OtherMetricsStruct.mat'), 'OtherMetricsStruct');
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
     

