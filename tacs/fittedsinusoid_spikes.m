% % %% ===== INTERACTIVE RAW+FITS + CLICKABLE RASTER (ELECTRODE) + COLLAPSED RASTER =====
% % before running it first load the spike sorted nev file and name it NEV_spike and then load the raw NEV file during the stimulation and the sinusoid like the following: 
% data_base = '';
% % 
% load('sliding_sinusoid_fits.mat');   % loads sliding_results, fs, window_length_sec, step_size_sec
% % % % 
% nsx = openNSx([data_base '.ns6'], 'read');
% % then run the script, similar plotting in the artifact_removal_2afc.m
% % 
% 

%%
ch = 20;

% for the start of the recording
plot_start_sec = 5;
plot_dur_sec   = 35;




% % for the end of the recording
% for plotting the end of the data
% plot_dur_sec   = 10;   % seconds (last 30 s)
% plot_start_sec = (nsx.MetaTags.DataPoints / fs) - plot_dur_sec;
% % 

T_start = plot_start_sec;
T_rast  = plot_dur_sec;

%% ================== RAW + FITS PREP ==================
fs = nsx.MetaTags.SamplingFreq;

start_sample = round(plot_start_sec * fs) + 1;
end_sample   = start_sample + round(plot_dur_sec * fs) - 1;

signal = double(nsx.Data(ch, start_sample:end_sample));
time   = (0:numel(signal)-1) / fs + plot_start_sec;

fit_time = [sliding_results.start_time];
idx_windows = find( ...
    fit_time + window_length_sec >= plot_start_sec & ...
    fit_time <= plot_start_sec + plot_dur_sec );

%% ================== SPIKES PREP ==================
S = NEV_spike.Data.Spikes;

% Raw fields
ts   = double(S.TimeStamp(:));
elec = double(S.Electrode(:));
unit = double(S.Unit(:));
wf   = double(S.Waveform);  % [nSamp x N]

% Convert timestamps to seconds
tsec = ts ./ double(NEV_spike.MetaTags.TimeRes);

% Keep only valid sorted units (exclude 0 and 255) ----
valid = (unit ~= 0) & (unit ~= 255);

% valid = (unit ~= 255);


ts   = ts(valid);
tsec = tsec(valid);
elec = elec(valid);
unit = unit(valid);
wf   = wf(:, valid);

% Waveform axis
nSamp = size(wf,1);
x_wf  = 1:nSamp;

idx_win   = (tsec >= T_start) & (tsec <= (T_start + T_rast));
spkIdx_all = find(idx_win);          % indices into the *filtered* arrays above
t_win_abs  = tsec(idx_win);          % absolute time (s)
t_win      = t_win_abs - T_start;    % 0..T_rast
e_win      = elec(idx_win);          % electrode per spike in window

elecs = unique(e_win);
nE = numel(elecs);

% Map electrode -> row index (1..nE)
[~, rowIdx] = ismember(e_win, elecs);

ord = sortrows([(1:numel(t_win_abs))' rowIdx t_win_abs], [2 3]);
ord = ord(:,1);

review_spkIdx = spkIdx_all(ord);
review_twin   = t_win(ord);
review_row    = rowIdx(ord);
review_elec   = e_win(ord);

nReview = numel(review_spkIdx);
if nReview == 0
    error('No spikes in the selected window (%.1f–%.1f s).', T_start, T_start + T_rast);
end

mainFig = figure('Name','Raw+Fits + Spike Raster (click spike to review)', 'NumberTitle','off');

ax1 = subplot(3,1,1); hold(ax1,'on');
ax2 = subplot(3,1,2); hold(ax2,'on');
ax3 = subplot(3,1,3); hold(ax3,'on');

plot(ax1, time - plot_start_sec, signal, 'k');

for w = idx_windows
    fit = sliding_results(w).fit;
    if isempty(fit) || any(~isfinite(fit)), continue; end

    fit_start_sample = round(sliding_results(w).start_time * fs) + 1;
    fit_end_sample   = fit_start_sample + numel(fit) - 1;

    a = max(start_sample, fit_start_sample);
    b = min(end_sample,   fit_end_sample);
    if a >= b, continue; end

    fit_idx  = (a:b) - fit_start_sample + 1;
    plot_idx = (a:b) - start_sample + 1;

    plot(ax1, (time(plot_idx) - plot_start_sec), fit(fit_idx), 'r');
end

xlim(ax1,[0 T_rast]);
ylabel(ax1,'Amplitude');
title(ax1, sprintf('Ch %d: raw + sliding sinusoid fits (%.1f–%.1f s)', ch, T_start, T_start+T_rast));
grid(ax1,'on');
legend(ax1, {'Raw','Sliding fits'}, 'Location','best');

baseColor = repmat([0 0 0], nReview, 1);   % black
markerSizes = 24 * ones(nReview,1);

rasterH = scatter(ax2, review_twin, review_row, markerSizes, baseColor, '|', ...
    'LineWidth', 1.2, 'PickableParts','all', 'HitTest','on');

xlim(ax2,[0 T_rast]);
ylim(ax2,[0.5 nE+0.5]);
yticks(ax2, 1:nE);
yticklabels(ax2, string(elecs));
ylabel(ax2,'Electrode');
title(ax2, 'Click a spike to review it in the waveform window');
grid(ax2,'on');

allH = scatter(ax3, review_twin, ones(nReview,1), markerSizes, baseColor, '|', 'LineWidth', 1.2);

xlim(ax3,[0 T_rast]);
ylim(ax3,[0.5 1.5]);
xlabel(ax3,'Time in window (s)');
ylabel(ax3,'All spikes');
title(ax3,'All spikes collapsed');
grid(ax3,'on');

linkaxes([ax1 ax2 ax3],'x');


excluded = false(nReview,1);

currentK = 1;

hl = plot(ax2, review_twin(currentK), review_row(currentK), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
uistack(hl,'top');



mainFig.UserData = struct( ...
    'wf', wf, ...
    'x_wf', x_wf, ...
    'tsec', tsec, ...
    'review_spkIdx', review_spkIdx, ...
    'review_twin', review_twin, ...
    'review_row', review_row, ...
    'review_elec', review_elec, ...
    'excluded', excluded, ...
    'currentK', currentK, ...
    'rasterH', rasterH, ...
    'allH', allH, ...                 % ADDed THIS
    'hl', hl, ...
    'T_start', T_start, ...
    'T_rast', T_rast);



rasterH.ButtonDownFcn = @(src,evt)onRasterClick(mainFig, evt);

waveFig = figure('Name','Waveform Review', 'NumberTitle','off');
waveAx  = axes('Parent', waveFig); hold(waveAx,'on'); grid(waveAx,'on');
xlabel(waveAx,'Waveform sample');
ylabel(waveAx,'Amplitude (raw)');

% Buttons
uicontrol(waveFig, 'Style','pushbutton', 'String','Prev', ...
    'Units','normalized', 'Position',[0.05 0.92 0.12 0.06], ...
    'Callback', @(~,~)stepSpike(mainFig, waveFig, -1));

uicontrol(waveFig, 'Style','pushbutton', 'String','Next', ...
    'Units','normalized', 'Position',[0.18 0.92 0.12 0.06], ...
    'Callback', @(~,~)stepSpike(mainFig, waveFig, +1));

uicontrol(waveFig, 'Style','togglebutton', 'String','Exclude', ...
    'Units','normalized', 'Position',[0.33 0.92 0.16 0.06], ...
    'Callback', @(btn,~)toggleExclude(mainFig, waveFig, btn));

uicontrol(waveFig, 'Style','pushbutton', 'String','Export excluded idx', ...
    'Units','normalized', 'Position',[0.51 0.92 0.22 0.06], ...
    'Callback', @(~,~)exportExcluded(mainFig));

uicontrol(waveFig, 'Style','pushbutton', 'String','Close', ...
    'Units','normalized', 'Position',[0.75 0.92 0.20 0.06], ...
    'Callback', @(~,~)close(waveFig));

% Initial plot
updateWaveformPlot(mainFig, waveFig);

function onRasterClick(mainFig, evt)
    % Select clicked spike from raster
    D = mainFig.UserData;

    % Try DataIndex first (best); fallback to nearest
    k = [];
    if isprop(evt, 'DataIndex') && ~isempty(evt.DataIndex)
        k = evt.DataIndex;
    end

    if isempty(k)
        % fallback: nearest point
        xClick = evt.IntersectionPoint(1);
        yClick = evt.IntersectionPoint(2);
        [~,k] = min((D.review_twin - xClick).^2 + (D.review_row - yClick).^2);
    end

    D.currentK = k;
    mainFig.UserData = D;

    % Update highlight + waveform window if open
    updateSelectionVisual(mainFig);
    waveFig = findobj(0,'Type','figure','Name','Waveform Review');
    if ~isempty(waveFig)
        updateWaveformPlot(mainFig, waveFig);
    end
end

function stepSpike(mainFig, waveFig, step)
    D = mainFig.UserData;
    k = D.currentK + step;
    k = max(1, min(numel(D.review_spkIdx), k));
    D.currentK = k;
    mainFig.UserData = D;

    updateSelectionVisual(mainFig);
    updateWaveformPlot(mainFig, waveFig);
end

function toggleExclude(mainFig, waveFig, btn)
    D = mainFig.UserData;
    k = D.currentK;

    % Toggle excluded status
    D.excluded(k) = ~D.excluded(k);
    mainFig.UserData = D;

    % Update button label to match state
    if D.excluded(k)
        btn.Value = 1;
        btn.String = 'Excluded';
    else
        btn.Value = 0;
        btn.String = 'Exclude';
    end

    % Update raster colors
    recolorRaster(mainFig);

    % Update waveform title
    updateWaveformPlot(mainFig, waveFig);
end

function updateSelectionVisual(mainFig)
    D = mainFig.UserData;
    k = D.currentK;

    % Move highlight marker
    set(D.hl, 'XData', D.review_twin(k), 'YData', D.review_row(k));
    drawnow limitrate;
end

function updateWaveformPlot(mainFig, waveFig)
    D = mainFig.UserData;
    k = D.currentK;

    spkIndex = D.review_spkIdx(k);   % original spike index into wf columns
    e = D.review_elec(k);
    tAbs = D.tsec(spkIndex);

    ax = findobj(waveFig,'Type','axes');
    cla(ax);

    plot(ax, D.x_wf, D.wf(:, spkIndex), 'k', 'LineWidth', 1.5);

    % Update exclude toggle button state text
    exBtn = findobj(waveFig,'Style','togglebutton');
    if ~isempty(exBtn)
        if D.excluded(k)
            exBtn.Value = 1;
            exBtn.String = 'Excluded';
        else
            exBtn.Value = 0;
            exBtn.String = 'Exclude';
        end
    end

    status = "KEEP";
    if D.excluded(k), status = "EXCLUDED"; end

    title(ax, sprintf('K=%d/%d | %s | Electrode %d | SpikeIdx %d | t=%.6f s', ...
        k, numel(D.review_spkIdx), status, e, spkIndex, tAbs));

    grid(ax,'on');
    drawnow limitrate;
end


function recolorRaster(mainFig)
    D = mainFig.UserData;

    % Color rule:
    % kept = black, excluded = light gray
    C = repmat([0 0 0], numel(D.review_spkIdx), 1);
    C(D.excluded,:) = repmat([0.7 0.7 0.7], sum(D.excluded), 1);

    % Middle raster
    D.rasterH.CData = C;

    % Bottom "all spikes" plot
    if isfield(D,'allH') && isgraphics(D.allH)
        D.allH.CData = C;
    end

    mainFig.UserData = D;
    drawnow limitrate;
end


function exportExcluded(mainFig)
    D = mainFig.UserData;

    excluded_spkIdx = D.review_spkIdx(D.excluded);  % ORIGINAL indices
    excluded_spkIdx = unique(excluded_spkIdx(:));

    assignin('base', 'excludedSpikeIdx', excluded_spkIdx);
    fprintf('Exported %d excluded spikes to workspace variable: excludedSpikeIdx\n', numel(excluded_spkIdx));
end










