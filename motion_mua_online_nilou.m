clear; clc; close all;

%% SETTINGS 
rlims             = [0.2, 0.3]; 
bandpass_cut_freq = [300, 12000];
do_rectify        = true;
baseline_lims     = [-0.2, 0];
plotWin           = [-0.2, 0.3];
smooth_ms         = 5;
chan_sel          = 1; % 13 for earlier recordings

%%  CONDITION LISTS 
cond_side = [ ...
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, ...
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];

cond_is_congruent = [ ...
0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ...
0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1];

num_stim = numel(cond_side);
assert(numel(cond_is_congruent) == num_stim);

%% CONDITION LABELS & COLORS 
cond_names = { ...
    'Change RIGHT + Incongruent  [right 1st → left 2nd → change RIGHT]', ...
    'Change RIGHT + Congruent    [left 1st → right 2nd → change RIGHT]', ...
    'Change LEFT  + Incongruent  [left 1st → right 2nd → change LEFT]',  ...
    'Change LEFT  + Congruent    [right 1st → left 2nd → change LEFT]'};

cond_colors = [ ...
    0.10 0.45 0.85; ...
    0.05 0.20 0.60; ...
    0.85 0.33 0.10; ...
    0.60 0.15 0.05];

%% LOAD FILES 
[file, path] = uigetfile({'*.nev;*.ns6', 'NEV/NS6 Files (*.nev, *.ns6)'}, ...
                         'Select NEV or NS6 file');
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

saveDir = fullfile(path, [fileName '_MUA_figures']);
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

if iscell(NS6.Data)
    if numel(NS6.Data) == 2
        pause_duration = NS6.MetaTags.Timestamp(2) - size(NS6.Data{1,1}, 2);
        if pause_duration < 0
            NS6.Data = [NS6.Data{1,1}, NS6.Data{1,2}];
        else
            nChan0   = size(NS6.Data{1,1}, 1);
            NS6.Data = [NS6.Data{1,1}, zeros(nChan0, pause_duration), NS6.Data{1,2}];
        end
    else
        tmp = NS6.Data{1};
        for k = 2:numel(NS6.Data)
            tmp = [tmp, NS6.Data{k}];
        end
        NS6.Data = tmp;
    end
end

fs      = double(NS6.MetaTags.SamplingFreq);
nSamp   = size(NS6.Data, 2);
timeRes = double(NEV.MetaTags.TimeRes);

dw  = double(NEV.Data.SerialDigitalIO.UnparsedData(:)');
dwt = double(NEV.Data.SerialDigitalIO.TimeStamp(:)');

%%  FIELDTRIP CONTINUOUS STRUCTURE 
[tlabels, tmdat1] = sortChannels(NS6);

data_raw         = [];
data_raw.trial   = {double(tmdat1)};
data_raw.time    = {(0:nSamp-1)/fs};
data_raw.fsample = fs;
data_raw.label   = tlabels;

data_cont = ft_preprocessing([], data_raw);

%%  PARSE TRIAL SEQUENCES 
% Digital stream per trial: [digit1  digit2  128    128    64       response]
%                            ix-2    ix-1    ix     ix+1   ix+2     ix+3
seq_pos = strfind(dw, [128 128 64]);

good = true(size(seq_pos));
for i = 1:numel(seq_pos)
    ix = seq_pos(i);
    if ix < 3, good(i) = false; continue; end
    d1 = dw(ix-2); d2 = dw(ix-1);
    if d1 >= 11 || d2 >= 11 || d1 <= 0 || d2 <= 0
        good(i) = false;
    end
end
seq_pos = seq_pos(good);
if isempty(seq_pos), error('No valid sequences found.'); end

stim_id_all   = nan(numel(seq_pos), 1);
onset_spd_sec = nan(numel(seq_pos), 1);
resp_rel_sec  = nan(numel(seq_pos), 1);

for i = 1:numel(seq_pos)
    ix = seq_pos(i);
    stim_id_all(i)   = (dw(ix-2)-1)*10 + (dw(ix-1)-1) + 1;
    spd_tick         = dwt(ix+2);
    onset_spd_sec(i) = spd_tick / timeRes;
    if ix+3 <= numel(dwt)
        resp_rel_sec(i) = (dwt(ix+3) - spd_tick) / timeRes;
    end
end

valid         = stim_id_all >= 1 & stim_id_all <= num_stim;
seq_pos       = seq_pos(valid);
stim_id_all   = stim_id_all(valid);
onset_spd_sec = onset_spd_sec(valid);
resp_rel_sec  = resp_rel_sec(valid);

side_all = cond_side(stim_id_all);
cong_all = cond_is_congruent(stim_id_all);

cond_code = (1 - side_all)*2 + cong_all + 1;

onset_stim_samp = nan(numel(seq_pos), 1);
onset_spd_samp  = nan(numel(seq_pos), 1);

right_is_second = (side_all == 1 & cong_all == 1) | ...
                  (side_all == 0 & cong_all == 0);

for i = 1:numel(seq_pos)
    ix = seq_pos(i);
    if right_is_second(i)
        onset_stim_samp(i) = round(dwt(ix+1) / timeRes * fs) + 1;
    else
        onset_stim_samp(i) = round(dwt(ix)   / timeRes * fs) + 1;
    end
    onset_spd_samp(i) = round(dwt(ix+2) / timeRes * fs) + 1;
end

%% RESPONSE TIME SUMMARY 
valid_resp      = ~isnan(resp_rel_sec) & resp_rel_sec > 0 & resp_rel_sec < 10;
resp_clean      = resp_rel_sec(valid_resp);

resp_median     = median(resp_clean);
resp_q1         = prctile(resp_clean, 25);
resp_q3         = prctile(resp_clean, 75);

fprintf('Parsed %d valid trials.\n', numel(seq_pos));
fprintf('Response time relative to speed change (n=%d trials):\n', numel(resp_clean));
fprintf('  Median : %.3f s\n', resp_median);
fprintf('  IQR    : [%.3f  %.3f] s\n', resp_q1, resp_q3);
for c = 1:4
    fprintf('  %s: %d\n', cond_names{c}, sum(cond_code==c));
end

%%  RUN BOTH ANALYSES 
analyses(1).label        = 'StimOnset_RightRF';
analyses(1).xlabel       = 'Time from stimulus onset in right RF (s)';
analyses(1).onset_samp   = onset_stim_samp;
analyses(1).show_resp    = false;

analyses(2).label        = 'SpeedChange';
analyses(2).xlabel       = 'Time from speed change (s)';
analyses(2).onset_samp   = onset_spd_samp;
analyses(2).show_resp    = true;

for an = 1:numel(analyses)

    onset_samp  = analyses(an).onset_samp;
    an_label    = analyses(an).label;
    an_xlabel   = analyses(an).xlabel;
    show_resp   = analyses(an).show_resp;
    cc          = cond_code;

    trl      = zeros(numel(onset_samp), 3);
    trl(:,1) = round(onset_samp - rlims(1)*fs);
    trl(:,2) = round(onset_samp + rlims(2)*fs);
    trl(:,3) = -round(rlims(1)*fs);
    trl(:,1) = max(trl(:,1), 1);
    trl(:,2) = min(trl(:,2), nSamp);

    bad = trl(:,2) <= trl(:,1);
    trl = trl(~bad,:);
    cc  = cc(~bad);

    cfg     = [];
    cfg.trl = trl;
    trial_data = ft_redefinetrial(cfg, data_cont);

    cfg          = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq   = bandpass_cut_freq;
    if do_rectify, cfg.rectify = 'yes'; end
    muadata_full = ft_preprocessing(cfg, trial_data);

%     cfg        = [];
%     cfg.method = 'summary';
%     cfg.box    = 'yes';
%     muadata_clean = ft_rejectvisual(cfg, muadata_full);
%
%     kept = muadata_clean.cfg.trials;
%     cc   = cc(kept);
%
%     fprintf('[%s] Kept %d / %d trials after rejection.\n', ...
%         an_label, numel(kept), numel(muadata_full.trial));

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
    zMask        = timevec >= baseline_lims(1) & timevec <= baseline_lims(2);

    %%  SMOOTHING KERNEL 
    smooth_samp = max(1, round(smooth_ms/1000 * fs));
    doSmooth    = smooth_samp > 1;
    if doSmooth
        g_len = 6 * smooth_samp + 1;   % odd length
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

    %%  Z-SCORE AFTER SMOOTHING 
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

    for ch = 1:num_channels

        fig = figure('Color','w');
        hold on;

        %% response time overlay
        resp_h = gobjects(0);
        if show_resp && resp_median >= plotWin(1) && resp_median <= plotWin(2)
            ylims_placeholder = [-0.3 0.5];
            patch([resp_q1 resp_q3 resp_q3 resp_q1], ...
                  [ylims_placeholder(1) ylims_placeholder(1) ...
                   ylims_placeholder(2) ylims_placeholder(2)], ...
                  [0.6 0.6 0.6], 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
                  'HandleVisibility', 'off');

            resp_h = xline(resp_median, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, ...
                  'DisplayName', sprintf('Resp median=%.0f ms (IQR [%.0f–%.0f])', ...
                  resp_median*1000, resp_q1*1000, resp_q3*1000));
        end

        lineHandles = gobjects(0);
        legLabels   = {};

        for c = 1:4
            trList = find(cc == c);
            if isempty(trList), continue; end

            X = zeros(numel(trList), numel(timevec));
            for k = 1:numel(trList)
                X(k,:) = muadata_clean.trial{trList(k)}(ch,:);
            end

            mu  = mean(X, 1, 'omitnan');
            sem = std(X,  0, 1, 'omitnan') ./ sqrt(size(X,1));

            col = cond_colors(c,:);

            p = fill([timevec fliplr(timevec)], ...
                     [mu+sem  fliplr(mu-sem)], ...
                     col, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            set(p, 'HandleVisibility', 'off');

            lh = plot(timevec, mu, 'LineWidth', 2, 'Color', col);
            lineHandles(end+1) = lh;
            legLabels{end+1}   = sprintf('%s  (n=%d)', cond_names{c}, numel(trList));
        end

        xline(0, '--k', 'LineWidth', 1.5);

        %% update IQR patch to match actual y limits
        if show_resp && resp_median >= plotWin(1) && resp_median <= plotWin(2)
            yl = ylim();
            patch_h = findobj(gca, 'Type', 'patch', 'FaceColor', [0.6 0.6 0.6]);
            if ~isempty(patch_h)
                for ph = 1:numel(patch_h)
                    patch_h(ph).Vertices(:,2) = [yl(1); yl(1); yl(2); yl(2)];
                end
            end
        end

        xlim(plotWin);
        grid on; box off;
        xlabel(an_xlabel);
        ylabel('MUA (Z-score)');
        title(sprintf('%s | %s | %s', fileName, muadata_clean.label{ch}, an_label), ...
              'Interpreter', 'none');

        if show_resp && ~isempty(resp_h) && isgraphics(resp_h)
            legend([lineHandles, resp_h], ...
                   [legLabels, {sprintf('Response median = %.0f ms  IQR [%.0f – %.0f ms]', ...
                   resp_median*1000, resp_q1*1000, resp_q3*1000)}], ...
                   'Location', 'best', 'FontSize', 9);
        else
            legend(lineHandles, legLabels, 'Location', 'best', 'FontSize', 9);
        end

        set(gca, 'FontSize', 13, 'FontWeight', 'bold');

        saveas(fig, fullfile(saveDir, ...
            sprintf('%s_%s_%s_MUA.png', fileName, an_label, muadata_clean.label{ch})));
    end
end

fprintf('\nDone. All figures saved to:\n  %s\n', saveDir);

%%  FUNCTION 
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