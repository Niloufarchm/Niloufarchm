clear; clc; close all;

rlims             = [0.15, 0.15];
bandpass_cut_freq = [300, 12000];
do_rectify        = true;

Stim_on  = 64;
Stim_off = 128;

plotWin       = [-0.15, 0.15];
baseline_lims = [-0.15, 0];
smooth_ms     = 5;

chan_sel = 1; % for the earlier sessions that the file was multi electrode for example 20251030 should be 13, othere: 1

%%  STIMULUS LISTS 
list_LGN_ID_hue = [ ...
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, ...
180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180, ...
270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270, ...
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];

list_LGN_ID_orientation = [ ...
0,0,0,0,0,45,45,45,45,45,90,90,90,90,90,135,135,135,135,135, ...
0,0,0,0,0,45,45,45,45,45,90,90,90,90,90,135,135,135,135,135, ...
0,0,0,0,0,45,45,45,45,45,90,90,90,90,90,135,135,135,135,135, ...
0,0,0,0,0,45,45,45,45,45,90,90,90,90,90,135,135,135,135,135];

list_LGN_ID_color_intensity = [ ...
3,7,11,15,19,3,7,11,15,19,3,7,11,15,19,3,7,11,15,19, ...
3,7,11,15,19,3,7,11,15,19,3,7,11,15,19,3,7,11,15,19, ...
3,7,11,15,19,3,7,11,15,19,3,7,11,15,19,3,7,11,15,19, ...
0.6,0.7,0.8,0.9,1,0.6,0.7,0.8,0.9,1,0.6,0.7,0.8,0.9,1,0.6,0.7,0.8,0.9,1];

num_stim = numel(list_LGN_ID_hue);
assert(numel(list_LGN_ID_orientation)     == num_stim, 'orientation list length mismatch');
assert(numel(list_LGN_ID_color_intensity) == num_stim, 'intensity list length mismatch');

%%  LOAD FILES 
[file, path] = uigetfile({'*.nev;*.ns6','NEV and NS6 Files (*.nev, *.ns6)'},'Select a NEV or NS6 file');
if isequal(file,0), error('No file selected.'); end

fileAndPath = fullfile(path,file);
[~,fileName,ext] = fileparts(fileAndPath);

switch lower(ext)
    case '.nev'
        openNEV(fileAndPath,'read','nomat','nosave','overwrite');
        nsxFilename = fullfile(path,[fileName,'.ns6']);
        if ~isfile(nsxFilename), error('NS6 file not found next to NEV.'); end
        openNSx(nsxFilename,'read');
    case '.ns6'
        openNSx(fileAndPath,'read');
        nevFilename = fullfile(path,[fileName,'.nev']);
        if ~isfile(nevFilename), error('NEV file not found next to NS6.'); end
        openNEV(nevFilename,'read','nomat','nosave','overwrite');
    otherwise
        error('Selected file is not a NEV or NS6 file.');
end

saveDir = fullfile(path, [fileName '_MUA_figures']);
if ~exist(saveDir,'dir'), mkdir(saveDir); end

if iscell(NS6.Data)
    if numel(NS6.Data)==2
        pause_duration = NS6.MetaTags.Timestamp(2) - size(NS6.Data{1,1},2);
        if pause_duration < 0
            NS6.Data = [NS6.Data{1,1},NS6.Data{1,2}];
        else
            nChan0 = size(NS6.Data{1,1},1);
            NS6.Data = [NS6.Data{1,1},zeros(nChan0,pause_duration),NS6.Data{1,2}];
        end
    else
        tmp = NS6.Data{1};
        for k = 2:numel(NS6.Data)
            tmp = [tmp,NS6.Data{k}];
        end
        NS6.Data = tmp;
    end
end

fs    = double(NS6.MetaTags.SamplingFreq);
nSamp = size(NS6.Data,2);

dw  = double(NEV.Data.SerialDigitalIO.UnparsedData(:)');
dwt = double(NEV.Data.SerialDigitalIO.TimeStamp(:)');

%% PARSE TRIAL SEQUENCES 
secs = strfind(dw,[Stim_on Stim_off]);

good = true(size(secs));
for i = 1:numel(secs)
    if secs(i) < 3
        good(i) = false; 
        continue;
    end
    tt = dw(secs(i)-2:secs(i)-1);
    if any(tt >= 11)
        good(i) = false;
    end
end
secs = secs(good);
if isempty(secs), error('No valid sequences found.'); end

mat_trials = nan(numel(secs),3);
for i = 1:numel(secs)
    mat_trials(i,1) = dwt(secs(i));
    mat_trials(i,2) = dwt(secs(i)+1);
    mat_trials(i,3) = (dw(secs(i)-2)-1)*10 + (dw(secs(i)-1)-1) + 1;
end

validStim      = mat_trials(:,3) >= 1 & mat_trials(:,3) <= num_stim;
mat_trials     = mat_trials(validStim,:);
stim_id_trials = mat_trials(:,3);

hue_trials       = list_LGN_ID_hue(stim_id_trials);
orient_trials    = list_LGN_ID_orientation(stim_id_trials);
intensity_trials = list_LGN_ID_color_intensity(stim_id_trials);

%%  FIELDTRIP STRUCTURE 
[tlabels,tmdat1] = sortChannels(NS6);

data         = [];
data.trial   = {double(tmdat1)};
data.time    = {(0:nSamp-1)/fs};
data.fsample = fs;
data.label   = tlabels;

data_cont = ft_preprocessing([],data);

trl      = zeros(size(mat_trials,1),3);
trl(:,1) = round(mat_trials(:,1) - rlims(1)*fs);
trl(:,2) = round(mat_trials(:,1) + rlims(2)*fs);
trl(:,3) = round(-rlims(1)*fs);
trl(:,1) = max(trl(:,1),1);
trl(:,2) = min(trl(:,2),nSamp);

bad              = trl(:,2) <= trl(:,1);
trl              = trl(~bad,:);
hue_trials       = hue_trials(~bad);
orient_trials    = orient_trials(~bad);
intensity_trials = intensity_trials(~bad);

cfg     = [];
cfg.trl = trl;
trial_data = ft_redefinetrial(cfg, data_cont);

cfg           = [];
cfg.bpfilter  = 'yes';
cfg.bpfreq    = bandpass_cut_freq;
if do_rectify, cfg.rectify = 'yes'; end
muadata_full = ft_preprocessing(cfg, trial_data);

% cfg        = [];
% cfg.method = 'summary';
% cfg.box    = 'yes';
% muadata_clean = ft_rejectvisual(cfg, muadata_full);
%
% kept_trials      = muadata_clean.cfg.trials;
% hue_trials       = hue_trials(kept_trials);
% orient_trials    = orient_trials(kept_trials);
% intensity_trials = intensity_trials(kept_trials);
%
% fprintf('Kept %d / %d trials after manual rejection.\n', numel(kept_trials), numel(muadata_full.trial));

muadata_clean = muadata_full;

cfg         = [];
cfg.latency = plotWin;
muadata_clean = ft_selectdata(cfg, muadata_clean);

if ~isempty(chan_sel)
    cfg         = [];
    cfg.channel = muadata_clean.label(chan_sel);
    muadata_clean = ft_selectdata(cfg, muadata_clean);
end

timevec      = muadata_clean.time{1};
num_channels = numel(muadata_clean.label);
num_trials   = numel(muadata_clean.trial);

%% SMOOTHING KERNEL 
smooth_samp = max(1, round(smooth_ms/1000 * fs));
doSmooth    = smooth_samp > 1;

if doSmooth
    g_len = 6 * smooth_samp + 1;   % force odd length
    g     = gausswin(g_len);
    g     = g / sum(g);
else
    g = 1;
end

%%  SMOOTH EACH TRIAL FIRST 
if doSmooth
    for ch = 1:num_channels
        for tr = 1:num_trials
            x = muadata_clean.trial{tr}(ch,:);
            muadata_clean.trial{tr}(ch,:) = smooth_with_mirror_padding(x, g);
        end
    end
end

%% Z-SCORE AFTER SMOOTHING 
zMask = timevec >= baseline_lims(1) & timevec <= baseline_lims(2);
if ~any(zMask)
    error('No samples in z-score baseline window.');
end

for ch = 1:num_channels
    baseline_cells = cell(num_trials,1);
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

%% ================================================================
%  ANALYSIS 1 — HUE
%% ================================================================
hueColor = containers.Map('KeyType','double','ValueType','any');
hueColor(-1)  = [0.10 0.10 0.10];
hueColor(0)   = [0.85 0.10 0.10];
hueColor(90)  = [0.95 0.70 0.10];
hueColor(180) = [0.10 0.60 0.10];
hueColor(270) = [0.10 0.30 0.85];

desired_hue_order = [-1 0 90 180 270];
present_hues      = unique(hue_trials(:))';
x_hue             = desired_hue_order(ismember(desired_hue_order, present_hues));

for ch = 1:num_channels
    fig = figure('Color','w'); hold on;
    lineHandles = gobjects(0);
    legLabels   = {};

    for i = 1:numel(x_hue)
        h      = x_hue(i);
        trList = find(hue_trials == h);
        if isempty(trList), continue; end

        X = zeros(numel(trList), numel(timevec));
        for k = 1:numel(trList)
            X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
        end

        mu  = mean(X,1,'omitnan');
        sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

        col = hueColor(h);
        p   = fill([timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                   col,'FaceAlpha',0.12,'EdgeColor','none');
        set(p,'HandleVisibility','off');
        lh = plot(timevec, mu,'LineWidth',2,'Color',col);
        lineHandles(end+1) = lh;
        legLabels{end+1}   = sprintf('%s  (n=%d)', hueLabel(h), numel(trList));
    end

    xline(0,'--k','LineWidth',1);
    xlim(plotWin); grid on; box off;
    xlabel('Time from onset (s)');
    ylabel('MUA (Z-score)');
    title(sprintf('%s | %s | MUA by Hue', fileName, muadata_clean.label{ch}), ...
          'Interpreter','none');
    placeLegend(lineHandles, legLabels);
    set(gca,'FontSize',13,'FontWeight','bold');
    saveas(fig, fullfile(saveDir, sprintf('%s_%s_Hue_MUA.png', fileName, muadata_clean.label{ch})));
end

%% ================================================================
%  ANALYSIS 2 — ORIENTATION
%% ================================================================
orient_vals   = unique(orient_trials(:))';
orient_colors = [ ...
    0.85 0.10 0.10; ...
    0.10 0.65 0.10; ...
    0.10 0.30 0.85; ...
    0.70 0.10 0.70];

orient_cmap = containers.Map('KeyType','double','ValueType','any');
for oi = 1:numel(orient_vals)
    orient_cmap(orient_vals(oi)) = orient_colors(mod(oi-1,size(orient_colors,1))+1,:);
end

for ch = 1:num_channels
    fig = figure('Color','w'); hold on;
    lineHandles = gobjects(0);
    legLabels   = {};

    for oi = 1:numel(orient_vals)
        ov     = orient_vals(oi);
        trList = find(orient_trials == ov);
        if isempty(trList), continue; end

        X = zeros(numel(trList), numel(timevec));
        for k = 1:numel(trList)
            X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
        end

        mu  = mean(X,1,'omitnan');
        sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

        col = orient_cmap(ov);
        p   = fill([timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                   col,'FaceAlpha',0.12,'EdgeColor','none');
        set(p,'HandleVisibility','off');
        lh = plot(timevec, mu,'LineWidth',2,'Color',col);
        lineHandles(end+1) = lh;
        legLabels{end+1}   = sprintf('%d°  (n=%d)', ov, numel(trList));
    end

    xline(0,'--k','LineWidth',1);
    xlim(plotWin); grid on; box off;
    xlabel('Time from onset (s)');
    ylabel('MUA (Z-score)');
    title(sprintf('%s | %s | MUA by Orientation', fileName, muadata_clean.label{ch}), ...
          'Interpreter','none');
    placeLegend(lineHandles, legLabels);
    set(gca,'FontSize',13,'FontWeight','bold');
    saveas(fig, fullfile(saveDir, sprintf('%s_%s_Orientation_MUA.png', fileName, muadata_clean.label{ch})));
end

%% ================================================================
%  ANALYSIS 3 — COLOR INTENSITY (all hues combined)
%% ================================================================
is_control   = hue_trials == -1;
is_chromatic = ~is_control;

chromatic_intensities = unique(intensity_trials(is_chromatic))';
control_intensities   = unique(intensity_trials(is_control))';

chrom_cmap   = cool(numel(chromatic_intensities));
control_cmap = hot(numel(control_intensities) + 2);
control_cmap = control_cmap(2:numel(control_intensities)+1,:);

for ch = 1:num_channels
    fig = figure('Color','w');

    ax1 = subplot(1,2,1); hold(ax1,'on');
    lineHandles1 = gobjects(0);
    legLabels1   = {};

    for ii = 1:numel(chromatic_intensities)
        iv     = chromatic_intensities(ii);
        trList = find(is_chromatic & intensity_trials == iv);
        if isempty(trList), continue; end

        X = zeros(numel(trList), numel(timevec));
        for k = 1:numel(trList)
            X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
        end

        mu  = mean(X,1,'omitnan');
        sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

        col = chrom_cmap(ii,:);
        p   = fill(ax1,[timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                   col,'FaceAlpha',0.12,'EdgeColor','none');
        set(p,'HandleVisibility','off');
        lh = plot(ax1, timevec, mu,'LineWidth',2,'Color',col);
        lineHandles1(end+1) = lh;
        legLabels1{end+1}   = sprintf('Intensity %g  (n=%d)', iv, numel(trList));
    end

    xline(ax1, 0,'--k','LineWidth',1);
    xlim(ax1, plotWin); grid(ax1,'on'); box(ax1,'off');
    xlabel(ax1,'Time from onset (s)');
    ylabel(ax1,'MUA (Z-score)');
    title(ax1,'Chromatic — all hues combined','FontSize',12);
    lg1 = legend(ax1, lineHandles1, legLabels1, 'Location','best', 'FontSize',10);
    setLegendTransparent(lg1);
    set(ax1,'FontSize',12,'FontWeight','bold');

    ax2 = subplot(1,2,2); hold(ax2,'on');
    lineHandles2 = gobjects(0);
    legLabels2   = {};

    for ii = 1:numel(control_intensities)
        iv     = control_intensities(ii);
        trList = find(is_control & intensity_trials == iv);
        if isempty(trList), continue; end

        X = zeros(numel(trList), numel(timevec));
        for k = 1:numel(trList)
            X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
        end

        mu  = mean(X,1,'omitnan');
        sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

        col = control_cmap(ii,:);
        p   = fill(ax2,[timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                   col,'FaceAlpha',0.12,'EdgeColor','none');
        set(p,'HandleVisibility','off');
        lh = plot(ax2, timevec, mu,'LineWidth',2,'Color',col);
        lineHandles2(end+1) = lh;
        legLabels2{end+1}   = sprintf('Luminance %g  (n=%d)', iv, numel(trList));
    end

    xline(ax2, 0,'--k','LineWidth',1);
    xlim(ax2, plotWin); grid(ax2,'on'); box(ax2,'off');
    xlabel(ax2,'Time from onset (s)');
    ylabel(ax2,'MUA (Z-score)');
    title(ax2,'Control (achromatic)','FontSize',12);
    lg2 = legend(ax2, lineHandles2, legLabels2, 'Location','best', 'FontSize',10);
    setLegendTransparent(lg2);
    set(ax2,'FontSize',12,'FontWeight','bold');

    sgtitle(sprintf('%s | %s | MUA by Intensity (all hues)', fileName, muadata_clean.label{ch}), ...
            'Interpreter','none','FontSize',13,'FontWeight','bold');

    saveas(fig, fullfile(saveDir, sprintf('%s_%s_Intensity_allHues_MUA.png', fileName, muadata_clean.label{ch})));
end

%% ================================================================
%  ANALYSIS 4 — INTENSITY SEPARATELY PER HUE
%% ================================================================
hue_base_color = containers.Map('KeyType','double','ValueType','any');
hue_base_color(0)   = [0.85 0.10 0.10];
hue_base_color(90)  = [0.95 0.70 0.10];
hue_base_color(180) = [0.10 0.60 0.10];
hue_base_color(270) = [0.10 0.30 0.85];
hue_base_color(-1)  = [0.20 0.20 0.20];

chromatic_hues = x_hue(x_hue ~= -1);

for ch = 1:num_channels

    for hi = 1:numel(chromatic_hues)
        h        = chromatic_hues(hi);
        hue_mask = hue_trials == h;
        iv_vals  = unique(intensity_trials(hue_mask))';
        if isempty(iv_vals), continue; end

        base_col  = hue_base_color(h);
        n_iv      = numel(iv_vals);
        shade_mat = zeros(n_iv, 3);
        for si = 1:n_iv
            frac = 0.25 + 0.75 * (si-1) / max(n_iv-1, 1);
            shade_mat(si,:) = min(1, base_col * frac + (1-frac)*[1 1 1]);
        end

        fig = figure('Color','w'); hold on;
        lineHandles = gobjects(0);
        legLabels   = {};

        for ii = 1:n_iv
            iv     = iv_vals(ii);
            trList = find(hue_mask & intensity_trials == iv);
            if isempty(trList), continue; end

            X = zeros(numel(trList), numel(timevec));
            for k = 1:numel(trList)
                X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
            end

            mu  = mean(X,1,'omitnan');
            sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

            col = shade_mat(ii,:);
            p   = fill([timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                       col,'FaceAlpha',0.15,'EdgeColor','none');
            set(p,'HandleVisibility','off');
            lh = plot(timevec, mu,'LineWidth',2,'Color',col);
            lineHandles(end+1) = lh;
            legLabels{end+1}   = sprintf('Intensity %g  (n=%d)', iv, numel(trList));
        end

        xline(0,'--k','LineWidth',1);
        xlim(plotWin); grid on; box off;
        xlabel('Time from onset (s)');
        ylabel('MUA (Z-score)');
        title(sprintf('%s | %s | %s — MUA by Intensity', ...
              fileName, muadata_clean.label{ch}, hueLabel(h)), 'Interpreter','none');
        placeLegend(lineHandles, legLabels);
        set(gca,'FontSize',13,'FontWeight','bold');

        hue_tag = regexprep(strrep(hueLabel(h),' ','_'), '[^a-zA-Z0-9_]', '');
        saveas(fig, fullfile(saveDir, ...
            sprintf('%s_%s_Intensity_%s_MUA.png', fileName, muadata_clean.label{ch}, hue_tag)));
    end

    ctrl_mask = hue_trials == -1;
    iv_vals   = unique(intensity_trials(ctrl_mask))';

    if ~isempty(iv_vals)
        base_col  = hue_base_color(-1);
        n_iv      = numel(iv_vals);
        shade_mat = zeros(n_iv, 3);
        for si = 1:n_iv
            frac = 0.25 + 0.75 * (si-1) / max(n_iv-1, 1);
            shade_mat(si,:) = min(1, base_col * frac + (1-frac)*[1 1 1]);
        end

        fig = figure('Color','w'); hold on;
        lineHandles = gobjects(0);
        legLabels   = {};

        for ii = 1:n_iv
            iv     = iv_vals(ii);
            trList = find(ctrl_mask & intensity_trials == iv);
            if isempty(trList), continue; end

            X = zeros(numel(trList), numel(timevec));
            for k = 1:numel(trList)
                X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
            end

            mu  = mean(X,1,'omitnan');
            sem = std(X,0,1,'omitnan') ./ sqrt(size(X,1));

            col = shade_mat(ii,:);
            p   = fill([timevec fliplr(timevec)],[mu+sem fliplr(mu-sem)], ...
                       col,'FaceAlpha',0.15,'EdgeColor','none');
            set(p,'HandleVisibility','off');
            lh = plot(timevec, mu,'LineWidth',2,'Color',col);
            lineHandles(end+1) = lh;
            legLabels{end+1}   = sprintf('Luminance %g  (n=%d)', iv, numel(trList));
        end

        xline(0,'--k','LineWidth',1);
        xlim(plotWin); grid on; box off;
        xlabel('Time from onset (s)');
        ylabel('MUA (Z-score)');
        title(sprintf('%s | %s | Control (achromatic) — MUA by Luminance', ...
              fileName, muadata_clean.label{ch}), 'Interpreter','none');
        placeLegend(lineHandles, legLabels);
        set(gca,'FontSize',13,'FontWeight','bold');

        saveas(fig, fullfile(saveDir, ...
            sprintf('%s_%s_Intensity_Control_MUA.png', fileName, muadata_clean.label{ch})));
    end
end

fprintf('\nDone. All figures saved to:\n  %s\n', saveDir);

%% ================================================================
%   FUNCTIONS 
%% ================================================================
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

function lbl = hueLabel(h)
switch h
    case -1,   lbl = 'Control (achromatic)';
    case 0,    lbl = 'Red (0°)';
    case 90,   lbl = 'Yellow (90°)';
    case 180,  lbl = 'Green (180°)';
    case 270,  lbl = 'Blue (270°)';
    otherwise, lbl = sprintf('Hue %g°', h);
end
end

function placeLegend(lineHandles, legLabels)
lg = legend(lineHandles, legLabels, ...
            'Location', 'best', ...
            'FontSize',  10);
setLegendTransparent(lg);
end

function setLegendTransparent(lg)
lg.EdgeColor = 'none';
try
    lg.BoxFace.ColorType = 'truecoloralpha';
    lg.BoxFace.ColorData = uint8([255; 255; 255; 128]);
catch
    lg.Color = [1 1 1];
end
end