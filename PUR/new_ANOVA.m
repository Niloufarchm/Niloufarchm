 % 1 **ANOVA POSITION ERROR EXCLUDING SACCADES 

load('PE_aggregated.mat');

vars = who;
posErrorTableVarName = vars{contains(vars, 'PE_aggregated')};
eval(['PE_aggregated = ' posErrorTableVarName ';']);

disp('Column names in the loaded table:');
disp(PE_aggregated.Properties.VariableNames);

PE_data = [PE_aggregated.PERight5, PE_aggregated.PERight10, PE_aggregated.PERight20, ...
           PE_aggregated.PELeft5, PE_aggregated.PELeft10, PE_aggregated.PELeft20];
PE_data = PE_data(:);

subjects = repmat((1:394)', [1, 6]); 
subjects = subjects(:); 

speeds = repmat([5, 10, 20, 5, 10, 20], [394, 1]); 
speeds = speeds(:); 

directions = repmat([1, 1, 1, 2, 2, 2], [394, 1]); % 1 for Right, 2 for Left, repeated for each subject
directions = directions(:); 

% Conduct two-way ANOVA
[p,tbl,stats] = anovan(PE_data, {speeds, directions}, 'model', 'interaction', ...
                       'varnames', {'Speed', 'Direction'}, 'random', 1);

disp(tbl);

% Post-hoc analysis using Tukey's HSD
[c,m,h,nms] = multcompare(stats,'Dimension',[1 2]);


directions_str = strings(size(directions));
directions_str(directions == 1) = 'ToRight';
directions_str(directions == 2) = 'ToLeft';

conditions = strcat(string(speeds), '_', string(directions_str));
[uniqueConditions, ~, condIdx] = unique(conditions, 'stable');

left_indices = find(directions == 2);
right_indices = find(directions == 1);

conditions_sorted = [conditions(left_indices); conditions(right_indices)];
PE_data_sorted = [PE_data(left_indices); PE_data(right_indices)];


figure;
boxplotHandle = boxplot(PE_data_sorted, conditions_sorted, 'Colors', 'k', 'Symbol', '', 'Notch', 'on');
set(boxplotHandle, 'LineWidth', 2); 
ylabel('Position Error');
xlabel('Condition (Speed & Direction)');
title('Position Error by Speed and Direction (Left 5, 10, 20; Right 5, 10, 20)');

set(gca, 'XTickLabelRotation', 45); 
hold on;

% Plotting means
means = zeros(length(uniqueConditions), 1);
errors = zeros(length(uniqueConditions), 1);

for i = 1:length(uniqueConditions)
    conditionData = PE_data_sorted(condIdx == i);
    means(i) = mean(conditionData);
    errors(i) = std(conditionData) / sqrt(sum(condIdx == i)); 
end

xPositions = linspace(1, length(uniqueConditions), length(uniqueConditions));
errorbar(xPositions, means, errors, 'k', 'linestyle', 'none', ...
         'LineWidth', 1.5, 'Marker', 's', 'MarkerSize', 4, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
plot(xPositions, means, 'ks', 'MarkerFaceColor', 'k'); % Plot means as black squares
hold off;



% 2) catchupsaccade
load('PUR_revised_data_aggregated.mat');


dataColumnNames = PUR_revised_data_aggregated.Properties.VariableNames(2:end);

dataMatrix = table2array(PUR_revised_data_aggregated(:, 2:end));

numSubjects = size(PUR_revised_data_aggregated, 1);
numConditions = numel(dataColumnNames); 

subjects = repmat((1:numSubjects)', [1, numConditions]); 
subjects = subjects(:); 

speeds = repmat([5, 10, 20, 5, 10, 20], [numSubjects, 1]);
speeds = speeds(:); 

directions = repmat([2, 2, 2, 1, 1, 1], [numSubjects, 1]); 
directions = directions(:); 

dataVector = dataMatrix(:);

% Conduct two-way ANOVA
[p,tbl,stats] = anovan(dataVector, {speeds, directions, subjects}, 'model', 'interaction', ...
                       'varnames', {'Speed', 'Direction', 'Subject'}, 'random', 3);

disp(tbl);

% Post-hoc analysis using Tukey's HSD
[c,m,h,nms] = multcompare(stats,'Dimension',[1 2]);

disp(tbl);


directions_str = strings(size(directions));
directions_str(directions == 1) = 'ToRight';
directions_str(directions == 2) = 'ToLeft';

conditions = strcat(string(speeds), '_', string(directions_str));
[uniqueConditions, ~, condIdx] = unique(conditions, 'stable');

figure;
boxplotHandle = boxplot(dataVector, conditions, 'Colors', 'k', 'Symbol', '', 'Notch', 'on');
set(boxplotHandle, 'LineWidth', 2); 
ylabel('Catch-up Saccade Metric');
xlabel('Condition (Speed & Direction)');
title('Catch-up Saccade by Speed and Direction');

set(gca, 'XTickLabelRotation', 45); 
hold on;

means = zeros(length(uniqueConditions), 1);
errors = zeros(length(uniqueConditions), 1);

for i = 1:length(uniqueConditions)
    conditionData = dataVector(condIdx == i);
    means(i) = mean(conditionData);
    errors(i) = std(conditionData) / sqrt(sum(condIdx == i)); 
end

% Plot means with error bars
xPositions = linspace(1, length(uniqueConditions), length(uniqueConditions));
errorbar(xPositions, means, errors, 'k', 'linestyle', 'none', ...
         'LineWidth', 1.5, 'Marker', 's', 'MarkerSize', 4, ...
         'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
hold off;







% 3)  TWO_WAY ANOVA velocity error
load('VelocityError_aggregated.mat');

vars = who;
VelocityErrorVarName = vars{contains(vars, 'VelocityError_aggregated')};
eval(['VelocityError_aggregated = ' VelocityErrorVarName ';']);

disp('Column names in the loaded table:');
disp(VelocityError_aggregated.Properties.VariableNames);

VE_data = [VelocityError_aggregated.VERight5, VelocityError_aggregated.VERight10, VelocityError_aggregated.VERight20, ...
           VelocityError_aggregated.VELeft5, VelocityError_aggregated.VELeft10, VelocityError_aggregated.VELeft20];
VE_data = VE_data(:); 

subjects = repmat((1:394)', [1, 6]); 
subjects = subjects(:);

speeds = repmat([5, 10, 20, 5, 10, 20], [394, 1]); 
speeds = speeds(:); 

directions = repmat([1, 1, 1, 2, 2, 2], [394, 1]); 
directions = directions(:); 

[p,tbl,stats] = anovan(VE_data, {speeds, directions}, 'model', 'interaction', ...
                       'varnames', {'Speed', 'Direction'}, 'random', 1);

disp(tbl);

[c,m,h,nms] = multcompare(stats,'Dimension',[1 2]);


directions_str = strings(size(directions));
directions_str(directions == 1) = 'ToRight';
directions_str(directions == 2) = 'ToLeft';

conditions = strcat(string(speeds), '_', string(directions_str));
[uniqueConditions, ~, condIdx] = unique(conditions, 'stable');

left_indices = find(directions == 2);
right_indices = find(directions == 1);

conditions_sorted = [conditions(left_indices); conditions(right_indices)];
VE_data_sorted = [VE_data(left_indices); VE_data(right_indices)];

figure;
boxplotHandle = boxplot(VE_data_sorted, conditions_sorted, 'Colors', 'k', 'Symbol', '', 'Notch', 'on');
set(boxplotHandle, 'LineWidth', 2); 
ylabel('Velocity Error');
xlabel('Condition (Speed & Direction)');
title('Velocity Error by Speed and Direction (Left 5, 10, 20; Right 5, 10, 20)');

set(gca, 'XTickLabelRotation', 45); 
hold on;

% Compute means and errors
means = zeros(length(uniqueConditions), 1);
errors = zeros(length(uniqueConditions), 1);

for i = 1:length(uniqueConditions)
    conditionData = VE_data_sorted(condIdx == i);
    means(i) = mean(conditionData);
    errors(i) = std(conditionData) / sqrt(sum(condIdx == i)); 
end

xPositions = linspace(1, length(uniqueConditions), length(uniqueConditions));
errorbar(xPositions, means, errors, 'k', 'linestyle', 'none', ...
         'LineWidth', 1.5, 'Marker', 's', 'MarkerSize', 4, ...
         'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
hold off;




