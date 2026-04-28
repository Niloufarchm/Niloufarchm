clear; clc; close all;

Stim_on  = 64;
Stim_off = 128;

rlims = [0.2 0.2];
bandpass_cut_freq = [300 12000];
do_rectify = true;

baseline_lims = [-0.2 0];
chan_sel = 1;

StartTime = 0.02;
EndTime   = 0.08;

smooth_ms = 5;

stim_list_x = [-4,-4,-4,-4,-4,-4,-4,-4,-4,-3,-3,-3,-3,-3,-3,-3,-3,-3,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4];
stim_list_y = [-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4,-4,-3,-2,-1,0,1,2,3,4];

xcenter=17.5; % 17.5  ,for the foveal: 5, superfoveal: 3
ycenter=0;  % 0
xscaling=3.888; %3.888,   for the foveal: 1.111,   0.3333 superfovel
yscaling=4.444; %4.444,   for the foveal: 1.111,   0.3333 superfoveal

stim_list_x = stim_list_x*xscaling + xcenter;
stim_list_y = stim_list_y*yscaling + ycenter;

num_stim = numel(stim_list_x);

[file, path] = uigetfile({'*.nev;*.ns6', 'NEV/NS6 Files (*.nev, *.ns6)'}, ...
                         'Select perimetry NEV or NS6');
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

if iscell(NS6.Data)
    tmp = NS6.Data{1};
    for k = 2:numel(NS6.Data)
        tmp = [tmp, NS6.Data{k}];
    end
    NS6.Data = tmp;
end

raw = double(NS6.Data);
fs  = double(NS6.MetaTags.SamplingFreq);
[nChan, nSamp] = size(raw);

assert(isfield(NEV,'Data') && isfield(NEV.Data,'SerialDigitalIO'), 'NEV SerialDigitalIO missing.');
dw  = double(NEV.Data.SerialDigitalIO.UnparsedData(:)');
dwt = double(NEV.Data.SerialDigitalIO.TimeStamp(:)');

assert(isfield(NEV.MetaTags,'TimeRes') && ~isempty(NEV.MetaTags.TimeRes), 'NEV.MetaTags.TimeRes missing.');
timeRes = double(NEV.MetaTags.TimeRes);

if isfield(NS6.MetaTags,'Timestamp') && ~isempty(NS6.MetaTags.Timestamp)
    nsx_start_tick = double(NS6.MetaTags.Timestamp);
else
    nsx_start_tick = 0;
end

secs = strfind(dw, [Stim_on Stim_off]);

good = true(size(secs));
for i = 1:numel(secs)
    if secs(i) < 3
        good(i) = false;
        continue;
    end
    tt = dw(secs(i)-2:secs(i)-1);
    if any(tt >= 11) || any(tt <= 0)
        good(i) = false;
    end
end
secs = secs(good);

if isempty(secs)
    error('No valid [Stim_on Stim_off]=[%d %d] sequences found.', Stim_on, Stim_off);
end

stimID  = nan(numel(secs),1);
on_tick = nan(numel(secs),1);

for i = 1:numel(secs)
    on_tick(i) = dwt(secs(i));
    stimID(i)  = (dw(secs(i)-2)-1).*10 + (dw(secs(i)-1)-1) + 1;
end

validStim = stimID >= 1 & stimID <= num_stim & ~isnan(on_tick);
stimID  = stimID(validStim);
on_tick = on_tick(validStim);

on_sec  = on_tick ./ timeRes;
on_samp = round((on_sec - nsx_start_tick/timeRes) * fs) + 1;

preS  = round(rlims(1)*fs);
postS = round(rlims(2)*fs);

good   = (on_samp - preS >= 1) & (on_samp + postS <= nSamp);
on_samp = on_samp(good);
stimID  = stimID(good);

fprintf('Parsed %d valid stimulus onsets.\n', numel(on_samp));

x_pos = stim_list_x(stimID);
y_pos = stim_list_y(stimID);

Wn = bandpass_cut_freq/(fs/2);
Wn(1) = max(Wn(1), 1e-6);
Wn(2) = min(Wn(2), 0.999999);

[b,a] = butter(2, Wn, 'bandpass');
filtSig = filtfilt(b, a, raw')';

if do_rectify
    mua = abs(filtSig);
else
    mua = filtSig;
end

if isempty(chan_sel)
    chan_sel = 1:nChan;
else
    chan_sel = chan_sel(:)';
    chan_sel = chan_sel(chan_sel >= 1 & chan_sel <= nChan);
    if isempty(chan_sel), error('chan_sel empty after bounds check.'); end
end

mua = mua(chan_sel, :);
chan_labels = chan_sel;
nChanSel = size(mua,1);

data_ft = [];
data_ft.fsample = fs;
data_ft.time    = {(0:nSamp-1)/fs};
data_ft.trial   = {mua};
for ch = 1:nChanSel
    data_ft.label{ch,1} = sprintf('ch%02d', chan_sel(ch));
end

trl = zeros(numel(on_samp), 3);
trl(:,1) = round(on_samp - rlims(1)*fs);
trl(:,2) = round(on_samp + rlims(2)*fs);
trl(:,3) = -preS;

trl(:,1) = max(trl(:,1), 1);
trl(:,2) = min(trl(:,2), nSamp);

bad = trl(:,2) <= trl(:,1);
trl    = trl(~bad,:);
x_pos  = x_pos(~bad);
y_pos  = y_pos(~bad);
stimID = stimID(~bad);

cfg     = [];
cfg.trl = trl;
muadata_full = ft_redefinetrial(cfg, data_ft);

% cfg        = [];
% cfg.method = 'summary';
% cfg.box    = 'yes';
% muadata_clean = ft_rejectvisual(cfg, muadata_full);
%
% kept_trials = muadata_clean.cfg.trials;
% x_pos  = x_pos(kept_trials);
% y_pos  = y_pos(kept_trials);
% stimID = stimID(kept_trials);

muadata_clean = muadata_full;

nTr = numel(muadata_clean.trial);
fprintf('Kept %d / %d trials after manual rejection.\n', nTr, numel(muadata_full.trial));

t = muadata_clean.time{1};
L = numel(t);

zMask = (t >= baseline_lims(1)) & (t <= baseline_lims(2));
if ~any(zMask)
    error('No samples in z-score baseline window.');
end

%% Gaussian kernel: force ODD length
smooth_samp = max(1, round((smooth_ms/1000) * fs));
doSmooth    = smooth_samp > 1;
if doSmooth
    g_len = 6*smooth_samp + 1;
    g = gausswin(g_len);
    g = g / sum(g);
else
    g = 1;
end

%% Smooth EACH trial first, then z-score
for ch = 1:nChanSel
    for tr = 1:nTr
        x = muadata_clean.trial{tr}(ch,:);
        if doSmooth
            muadata_clean.trial{tr}(ch,:) = smooth_with_mirror_padding(x, g);
        end
    end
end

%% Convert to ep array after smoothing
ep = nan(nChanSel, L, nTr);
for tr = 1:nTr
    ep(:,:,tr) = muadata_clean.trial{tr};
end

%% Z-score using baseline pooled across all trials, from SMOOTHED data
baseline = reshape(ep(:, zMask, :), nChanSel, []);
mu_z     = mean(baseline, 2, 'omitnan');
sd_z     = std(baseline, 0, 2, 'omitnan');
sd_z(sd_z == 0 | isnan(sd_z)) = 1;

ep = (ep - mu_z) ./ sd_z;

%% Put z-scored data back into muadata_clean so everything uses the same data
for tr = 1:nTr
    muadata_clean.trial{tr} = ep(:,:,tr);
end

maskWin = (t >= StartTime) & (t <= EndTime);
if ~any(maskWin)
    error('No samples in summary window [StartTime EndTime].');
end

tmp = mean(ep(:, maskWin, :), 2, 'omitnan');
trialResp = reshape(tmp, nChanSel, nTr);

x_vals = unique(x_pos);
y_vals = unique(y_pos);

rf_map = nan(numel(y_vals), numel(x_vals), nChanSel);
rf_n   = zeros(numel(y_vals), numel(x_vals), nChanSel);
trial_index = cell(numel(y_vals), numel(x_vals));

for tr = 1:nTr
    xi = find(x_vals == x_pos(tr), 1, 'first');
    yi = find(y_vals == y_pos(tr), 1, 'first');
    if isempty(xi) || isempty(yi), continue; end
    trial_index{yi,xi}(end+1) = tr;
end

for ch = 1:nChanSel
    for xi = 1:numel(x_vals)
        for yi = 1:numel(y_vals)
            idx = trial_index{yi,xi};
            if isempty(idx), continue; end
            rf_map(yi,xi,ch) = mean(trialResp(ch,idx), 'omitnan');
            rf_n(yi,xi,ch)   = numel(idx);
        end
    end
end

for ch = 1:nChanSel
    f = figure('Color','w');
    ax = axes('Parent', f);
    hImg = imagesc(ax, x_vals, y_vals, rf_map(:,:,ch));
    set(ax, 'YDir', 'normal');
    axis(ax, 'image');
    xlabel(ax, 'X position');
    ylabel(ax, 'Y position');
    title(ax, sprintf('%s | Ch %d | RF map', fileName, chan_labels(ch)), 'Interpreter', 'none');
    colorbar(ax);

    setappdata(f, 't', t);
    setappdata(f, 'ep', ep);                 % already smoothed + z-scored
    setappdata(f, 'trial_index', trial_index);
    setappdata(f, 'x_vals', x_vals);
    setappdata(f, 'y_vals', y_vals);
    setappdata(f, 'channel_index', ch);
    setappdata(f, 'fileName', fileName);

    set(hImg, 'PickableParts', 'all');
    set(hImg, 'ButtonDownFcn', @rf_click_callback);
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
end

for ch = 1:nChanSel
    [mx, idx] = max(rf_map(:,:,ch), [], 'all', 'omitnan');
    if ~isnan(mx)
        [yy, xx] = ind2sub(size(rf_map(:,:,ch)), idx);
        fprintf('Ch %d best response at x = %g, y = %g, mean MUA = %.4f\n', ...
            chan_labels(ch), x_vals(xx), y_vals(yy), mx);
    end
end

fprintf('Done.\n');

%% Global response per channel across all trials
global_mean_ch = mean(ep, 3, 'omitnan');
global_sem_ch  = std(ep, 0, 3, 'omitnan') ./ sqrt(nTr);

for ch = 1:nChanSel
    figure('Color','w');
    hold on;

    fill([t fliplr(t)], ...
         [global_mean_ch(ch,:)+global_sem_ch(ch,:) fliplr(global_mean_ch(ch,:)-global_sem_ch(ch,:))], ...
         [0 0 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    plot(t, global_mean_ch(ch,:), 'k', 'LineWidth', 2);
    xline(0, '--k', 'LineWidth', 1);

    xlabel('Time from onset (s)');
    ylabel('MUA (Z-score)');
    title(sprintf('%s | Ch %d | Global response across all trials (n=%d)', ...
          fileName, chan_labels(ch), nTr), 'Interpreter', 'none');
    xlim([-0.2 0.2]);
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    box off;
end

function rf_click_callback(src, ~)
    fig = ancestor(src, 'figure');
    ax  = ancestor(src, 'axes');

    t           = getappdata(fig, 't');
    ep          = getappdata(fig, 'ep');   % already smoothed + z-scored
    trial_index = getappdata(fig, 'trial_index');
    x_vals      = getappdata(fig, 'x_vals');
    y_vals      = getappdata(fig, 'y_vals');
    ch          = getappdata(fig, 'channel_index');
    fileName    = getappdata(fig, 'fileName');

    pt      = get(ax, 'CurrentPoint');
    x_click = pt(1,1);
    y_click = pt(1,2);

    [~, xi] = min(abs(x_vals - x_click));
    [~, yi] = min(abs(y_vals - y_click));

    idx = trial_index{yi,xi};
    if isempty(idx)
        disp('No trials here.');
        return;
    end

    X = squeeze(ep(ch,:,idx));
    if isvector(X)
        X = X(:);
    end
    if size(X,1) ~= numel(t)
        X = X';
    end

    m = mean(X, 2, 'omitnan')';
    s = std(X, 0, 2, 'omitnan')' ./ sqrt(size(X,2));

    figure('Color','w');
    hold on;
    fill([t fliplr(t)], [m+s fliplr(m-s)], [0 0 0], ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(t, m, 'k', 'LineWidth', 2);
    xline(0, '--k', 'LineWidth', 1);
    xlabel('Time from onset (s)');
    ylabel('MUA (Z-score)');
    title(sprintf('%s | x=%g y=%g | n=%d', fileName, x_vals(xi), y_vals(yi), numel(idx)), ...
        'Interpreter', 'none');
    xlim([-0.2 0.2]);
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    box off;
end

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

    left_part  = x(2:min(n, pad+1));
    right_part = x(max(1, n-pad):n-1);

    left_pad  = fliplr(left_part);
    right_pad = fliplr(right_part);

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