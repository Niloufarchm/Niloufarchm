    
% MICROSACCADE ENTROPY


folderPath = '';
targetFolderPath = '';
fileList = dir(fullfile(folderPath, '*.csv'));


if ~isempty(fileList)
    accumulateData(folderPath, targetFolderPath, fileList);
else
    disp('No CSV files found in the specified folder.');
end

function accumulateData(folderPath, targetFolderPath, fileList)
   
    TemporalEntropyMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

    
    
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

        if isempty(Trials.Microsaccades.StartTime)
            % Handle the case with no microsaccades: skip or set default values
            disp(['No microsaccades detected for ', fileName]);
            continue; % Skip this file or set default values for TemporalEntropyMap
        end


        

        
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
        

        microsaccadesStartTimes = Trials.Microsaccades.StartTime(microsaccadeIndices);
        microsaccadesEndTimes = Trials.Microsaccades.EndTime(microsaccadeIndices);
        latencies = zeros(1, length(microsaccadesStartTimes));
        if ~isempty(microsaccadesStartTimes)
            latencies(1) = microsaccadesStartTimes(1);
            for j = 2:length(microsaccadesStartTimes)
                latencies(j) = (microsaccadesStartTimes(j)) - (microsaccadesEndTimes(j-1));
            end
        end
                
        Duration = Trials.Samples.time(end); % ms
        
        % Normalize microsaccade start and end times to be between 0 and 1
        microsaccadesStartTimes = Trials.Microsaccades.StartTime(microsaccadeIndices) / Duration;
        microsaccadesEndTimes = Trials.Microsaccades.EndTime(microsaccadeIndices) / Duration;
        
        
     
        % Shannon entropy
        % Temporal entropy on normalized latencies
        latenciesPdf = latencies / sum(latencies);
        latenciesPdf(latenciesPdf == 0) = [];
        temporalEntropy = -sum(latenciesPdf .* log2(latenciesPdf));

        TemporalEntropyMap(subjectNumber) = temporalEntropy;
        
        
        TemporalEntropyMap(subjectNumber) = struct('microsaccadesStartTimes', microsaccadesStartTimes, 'latencies', latencies, 'TemporalEntropy', temporalEntropy);
        movefile(fullfile(folderPath, fileName), targetFolderPath);      
    end
    TemporalEntropyStruct = map2struct(TemporalEntropyMap);
    
    save(fullfile(targetFolderPath, 'TemporalEntropyStruct.mat'), 'TemporalEntropyStruct');
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
     

