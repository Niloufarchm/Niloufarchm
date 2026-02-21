% Calculate relative end coordinates
PUR_saccadesTable.relativeEndX = PUR_saccadesTable.EndX - PUR_saccadesTable.StartX;
PUR_saccadesTable.relativeEndY = PUR_saccadesTable.EndY - PUR_saccadesTable.StartY;

% Get unique combinations of Subject, Direction, and Speed
uniqueCombinations = unique(PUR_saccadesTable(:, {'Subject'}));

combinationList = {};  
totalEntropyValues = [];
meanEndXValues = [];
meanEndYValues = [];
stdEndXValues = [];
stdEndYValues = [];
averageAmplitudeValues = [];

% Define edges for histograms
numBins = 100;
xEdges = linspace(-10, 10, numBins + 1);
yEdges = linspace(-10, 10, numBins + 1);

for i = 1:height(uniqueCombinations)
    currentCombination = uniqueCombinations(i, :);
    
    condition = ismember(PUR_saccadesTable.Subject, currentCombination.Subject);
    combinationData = PUR_saccadesTable(condition, :);
    
    h = histogram2(combinationData.relativeEndX, combinationData.relativeEndY, xEdges, yEdges, ...
                   'DisplayStyle', 'tile', 'ShowEmptyBins', 'on', 'Normalization', 'probability', 'Visible', 'on');
    
    fig = gcf;  
    fig.Visible = 'on';
    filename = sprintf('histogram_%d.jpg', i);
    saveas(fig, filename);  
    
    img = imread(filename);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Normalize the image to range [0, 1]
    img = im2double(img);
    totalEntropyVal = entropy(img);
    
    meanEndXVal = nanmean(abs(combinationData.relativeEndX));
    meanEndYVal = nanmean(abs(combinationData.relativeEndY));
    stdEndXVal = nanstd(abs(combinationData.relativeEndX));
    stdEndYVal = nanstd(abs(combinationData.relativeEndY));
    averageAmplitudeVal = nanmean(combinationData.Amplitude);
    
    combinationList = [combinationList; {currentCombination.Subject{1}}];  
    totalEntropyValues(end+1, 1) = totalEntropyVal;
    meanEndXValues(end+1, 1) = meanEndXVal;
    meanEndYValues(end+1, 1) = meanEndYVal;
    stdEndXValues(end+1, 1) = stdEndXVal;
    stdEndYValues(end+1, 1) = stdEndYVal;
    averageAmplitudeValues(end+1, 1) = averageAmplitudeVal;
end

combinationTable = cell2table(combinationList, 'VariableNames', {'Subject'});

EntropyTable = [combinationTable, table(totalEntropyValues, meanEndXValues, meanEndYValues, stdEndXValues, stdEndYValues, averageAmplitudeValues, ...
                 'VariableNames', {'SpatialEntropy', 'MeanEndX', 'MeanEndY', 'StdEndX', 'StdEndY', 'AverageAmplitude'})];

