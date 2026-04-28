clear; clc; close all;

FLASH_ON_CODE     = 64;
rlims             = [0.15 0.2];          % [pre post] in seconds
bandpass_cut_freq = [300 12000];
baseline_lims     = [-0.15 0];           % for z-scoring
do_rectify        = true;
smooth_ms         = 5;                   % Gaussian smoothing width in ms

chan_sel          = 1;                   % channel(s) to analyse; [] = all, 13 for the earlier recordings

%% Pick file
[file, path] = uigetfile({'*.nev;*.ns6', 'NEV/NS6 Files (*.nev, *.ns6)'}, ...
                         'Select NEV or NS6 file (flash session)');
if isequal(file,0), error('No file selected.'); end
fileAndPath = fullfile(path, file);
[~, fileName, ext] = fileparts(fileAndPath);

switch lower(ext)
    case '.nev'
        openNEV(fileAndPath, 'read', 'nomat', 'nosave', 'overwrite');
        nsxFilename = fullfile(path, [fileName '.ns6']);
        if ~isfile(nsxFilename), error('NS6 not found next to NEV.'); end
        openNSx(nsxFilename, 'read');
    case '.ns6'
        openNSx(fileAndPath, 'read');
        nevFilename = fullfile(path, [fileName '.nev']);
        if ~isfile(nevFilename), error('NEV not found next to NS6.'); end
        openNEV(nevFilename, 'read', 'nomat', 'nosave', 'overwrite');
    otherwise
        error('Pick a .nev or .ns6 file.');
end

%% NS6 – flatten cell array if needed
if iscell(NS6.Data)
    tmp = NS6.Data{1};
    for k = 2:numel(NS6.Data)
        tmp = [tmp, NS6.Data{k}];
    end
    NS6.Data = tmp;
end
raw   = double(NS6.Data);
fs    = double(NS6.MetaTags.SamplingFreq);
[nChanRaw, nSamp] = size(raw);

%% Digital events → flash onset samples
dw  = double(NEV.Data.SerialDigitalIO.UnparsedData(:)');
dwt = double(NEV.Data.SerialDigitalIO.TimeStamp(:)');

assert(isfield(NEV.MetaTags,'TimeRes') && ~isempty(NEV.MetaTags.TimeRes));
timeRes = double(NEV.MetaTags.TimeRes);

if isfield(NS6.MetaTags,'Timestamp') && ~isempty(NS6.MetaTags.Timestamp)
    nsx_start_tick = double(NS6.MetaTags.Timestamp);
else
    nsx_start_tick = 0;
end

on_idx = find(dw == FLASH_ON_CODE);
if isempty(on_idx), error('No digital code %d found in NEV.', FLASH_ON_CODE); end

onset_sec  = dwt(on_idx) / timeRes;
onset_samp = round((onset_sec - nsx_start_tick/timeRes) * fs) + 1;

good       = onset_samp > 1 & onset_samp < nSamp;
onset_samp = onset_samp(good);
fprintf('Found %d flash onsets.\n', numel(onset_samp));

%% Bandpass filter + rectify → MUA
[b, a] = butter(2, bandpass_cut_freq / (fs/2), 'bandpass');
filt   = filtfilt(b, a, raw')';          % zero-phase filter
if do_rectify
    mua = abs(filt);
else
    mua = filt;
end

%% Channel selection
if isempty(chan_sel)
    chan_sel = 1:nChanRaw;
end
chan_sel = chan_sel(chan_sel >= 1 & chan_sel <= nChanRaw);
mua      = mua(chan_sel, :);
nChan    = size(mua, 1);

%% Build FieldTrip data structure (continuous)
data_ft           = [];
data_ft.fsample   = fs;
data_ft.time      = {(0:nSamp-1) / fs};
data_ft.trial     = {mua};
for ch = 1:nChan
    data_ft.label{ch,1} = sprintf('ch%02d', chan_sel(ch));
end

%% Build trl matrix
% col1 = start sample, col2 = end sample, col3 = offset
trl      = zeros(numel(onset_samp), 3);
trl(:,1) = round(onset_samp - rlims(1)*fs);
trl(:,2) = round(onset_samp + rlims(2)*fs);
trl(:,3) = -round(rlims(1)*fs);

% Remove trials outside recording
valid    = trl(:,1) >= 1 & trl(:,2) <= nSamp;
trl      = trl(valid, :);
fprintf('%d trials within recording bounds.\n', size(trl,1));

%% Segment continuous data into trials
cfg     = [];
cfg.trl = trl;
muadata_full = ft_redefinetrial(cfg, data_ft);

%% Manual trial rejection
% cfg        = [];
% cfg.method = 'summary';
% cfg.box    = 'yes';
% muadata_clean = ft_rejectvisual(cfg, muadata_full);
% 
% nTrialsKept  = numel(muadata_clean.trial);
% nTrialsTotal = numel(muadata_full.trial);
% fprintf('Kept %d / %d trials after manual rejection.\n', nTrialsKept, nTrialsTotal);

%% Use all kept trials
muadata_clean = muadata_full;
nTrialsKept   = numel(muadata_clean.trial);

%% Common time axis and baseline mask
t     = muadata_clean.time{1};
zMask = t >= baseline_lims(1) & t <= baseline_lims(2);

if ~any(zMask)
    error('Baseline mask is empty. Check baseline_lims and epoch window.');
end

%% Gaussian smoother kernel (ODD length only)
smooth_samp = max(1, round(smooth_ms/1000 * fs));
doSmooth    = smooth_samp > 1;

if doSmooth
    g_len = 6 * smooth_samp + 1;   % force odd kernel length
    g     = gausswin(g_len);
    g     = g / sum(g);
else
    g = 1;
end

%% Smooth EACH TRIAL first, then z-score
num_trials = numel(muadata_clean.trial);

if doSmooth
    for ch = 1:nChan
        for tr = 1:num_trials
            x = muadata_clean.trial{tr}(ch,:);
            muadata_clean.trial{tr}(ch,:) = smooth_with_mirror_padding(x, g);
        end
    end
end

%% Z-score using baseline pooled across all clean trials
for ch = 1:nChan
    baseline_cells = cell(num_trials, 1);
    for tr = 1:num_trials
        baseline_cells{tr} = muadata_clean.trial{tr}(ch, zMask);
    end
    baseline_all = [baseline_cells{:}];

    mu_z = mean(baseline_all, 'omitnan');
    sd_z = std(baseline_all, 0, 'omitnan');

    if sd_z == 0 || isnan(sd_z)
        sd_z = 1;
    end

    for tr = 1:num_trials
        muadata_clean.trial{tr}(ch,:) = ...
            (muadata_clean.trial{tr}(ch,:) - mu_z) / sd_z;
    end
end

%% Average across clean trials
allData_z = cat(3, muadata_clean.trial{:});     % nChan × nTime × nTrial
m         = mean(allData_z, 3, 'omitnan');      % nChan × nTime
s         = std(allData_z, 0, 3, 'omitnan') ./ sqrt(nTrialsKept);

%% Plot
for ch = 1:nChan
    figure('Color','w'); hold on;

    fill([t, fliplr(t)], ...
         [m(ch,:) + s(ch,:), fliplr(m(ch,:) - s(ch,:))], ...
         [0 0 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    plot(t, m(ch,:), 'k', 'LineWidth', 2);

    xline(0, '--k', 'LineWidth', 1);

    grid on;
    box off;
    xlabel('Time from flash onset (s)');
    ylabel('MUA (Z-scored)');
    title(sprintf('%s | Ch %d | Flash MUA (n=%d)', ...
          fileName, chan_sel(ch), nTrialsKept), ...
          'Interpreter', 'none');
    xlim([-rlims(1), rlims(2)]);
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
end

%% helper
function y = smooth_with_mirror_padding(x, g)
    x = double(x(:)');
    g = double(g(:)');

    L = numel(g);
    if mod(L,2) == 0
        error('Gaussian kernel length must be odd.');
    end

    pad = (L - 1) / 2;
    n   = numel(x);

    if n <= 1 || pad == 0
        y = x;
        return;
    end

    % Mirror padding excluding endpoint duplication
    left_part  = x(2:min(n, pad+1));
    right_part = x(max(1, n-pad):n-1);

    left_pad  = fliplr(left_part);
    right_pad = fliplr(right_part);

    % Top up if the signal is too short
    if numel(left_pad) < pad
        left_pad = [repmat(x(1), 1, pad-numel(left_pad)), left_pad];
    end
    if numel(right_pad) < pad
        right_pad = [right_pad, repmat(x(end), 1, pad-numel(right_pad))];
    end

    x_pad = [left_pad, x, right_pad];
    y_pad = conv(x_pad, g, 'same');
    y     = y_pad(pad+1 : pad+n);
end