%% Layered depth plot (NO Theil–Sen) + Layer summary with bootstrap reliability
% - Figure A: Δ vs depth with shaded layers, sig in red, nonsig dark gray, LOWESS only
% - Figure B: Layer-wise Δ summary with mean ± 95% bootstrap CI + n + %sig
% - All axes: bold, FontSize 16


alphaSig   = 0.05;
xlim_fixed = [-3 3];        
bootB      = 5000;         
rng(1);                    

granCh = struct();
granCh.session1       = [31 32];
granCh.session4       = [15 16 17 18];
granCh.session9       = [13 14 15];
granCh.session11      = [16 17 18];
granCh.session12      = [22 23 24];

granCh.control_Giu    = [4 5 6];
granCh.control_Cocoa  = [15 16 17 18];
granCh.control_Cocoa2 = [9 10 11 12];

granCh.session6       = [13 14 15]; % gamma
granCh.session7       = [8 9 10];   % gamma

nCh = struct();
nCh.session1       = 32;
nCh.session4       = 32;
nCh.session9       = 24;
nCh.session11      = 24;
nCh.session12      = 24;

nCh.control_Giu    = 24;
nCh.control_Cocoa  = 32;
nCh.control_Cocoa2 = 32;

nCh.session6       = 24;
nCh.session7       = 24;

spacing_um = struct();
spacing_um.ch24 = 100;
spacing_um.ch32 = 75;

cond = struct();
cond(1).label    = 'Alpha tACS';
cond(1).sessions = {'session1','session4','session9','session11','session12'};

cond(2).label    = 'Gamma tACS';
cond(2).sessions = {'session6','session7'};

cond(3).label    = 'Controls';
cond(3).sessions = {'control_Giu','control_Cocoa','control_Cocoa2'};

D = cell(3,1);
for k = 1:3
    D{k} = collect_condition_data(cond(k).sessions, granCh, nCh, spacing_um, alphaSig);
end

figure('Color','w','Name','Layered depth plots (LOWESS only)','Position',[80 80 1500 480]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for k = 1:3
    nexttile; hold on; grid on;
    plot_layered_depth_lowess(D{k}, cond(k).label, xlim_fixed);
    set(gca,'FontSize',18,'FontWeight','bold');
end

figure('Color','w','Name','Layer-wise summary (bootstrap CI)','Position',[100 120 1500 520]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for k = 1:3
    nexttile; hold on; grid on;
    plot_layer_summary_bootstrap(D{k}, cond(k).label, xlim_fixed, bootB, alphaSig);
    set(gca,'FontSize',18,'FontWeight','bold');
end


function S = collect_condition_data(sessList, granCh, nCh, spacing_um, alphaSig)
    

    deltaAll = [];
    depthAll = [];
    layerAll = [];
    sigAll   = false(0,1);

    gMinAll = +inf; gMaxAll = -inf;

    for i = 1:numel(sessList)
        sessName = sessList{i};

        if evalin('base', sprintf('exist(''%s'',''var'')', sessName)) ~= 1
            warning('Session table "%s" not found in workspace. Skipping.', sessName);
            continue;
        end

        T = evalin('base', sessName);

        [delta, depth_um, layerIdx, sigMask, gMin, gMax] = per_session_delta_depth_layer(T, sessName, granCh, nCh, spacing_um, alphaSig);

        deltaAll = [deltaAll; delta];
        depthAll = [depthAll; depth_um];
        layerAll = [layerAll; layerIdx];
        sigAll   = [sigAll; sigMask];

        gMinAll = min(gMinAll, gMin);
        gMaxAll = max(gMaxAll, gMax);
    end

    S = struct();
    S.delta   = deltaAll;
    S.depth_um= depthAll;
    S.layerIdx= layerAll;
    S.sigMask = sigAll;
    S.gMin    = gMinAll;
    S.gMax    = gMaxAll;
end

function [delta, depth_um, layerIdx, sigMask, gMin, gMax] = per_session_delta_depth_layer(T, sessName, granCh, nCh, spacing_um, alphaSig)

    if ~istable(T), error('%s is not a table.', sessName); end
    vars = T.Properties.VariableNames;

    if ~ismember('Electrode', vars)
        error('Table %s missing Electrode column.', sessName);
    end

    beforeName = vars(contains(vars, 'Before', 'IgnoreCase', true));
    afterName  = vars(contains(vars, 'After',  'IgnoreCase', true));
    if isempty(beforeName) || isempty(afterName)
        error('Table %s: no Before/After columns found.', sessName);
    end
    beforeName = beforeName{1};
    afterName  = afterName{1};

    elec = double(T.Electrode(:));
    beforeVal = double(T.(beforeName)); beforeVal = beforeVal(:);
    afterVal  = double(T.(afterName));  afterVal  = afterVal(:);
    delta = afterVal - beforeVal;

    % significance from p_FR
    if ismember('p_FR', vars)
        p = double(T.p_FR(:));
        sigMask = isfinite(p) & (p < alphaSig);
    else
        sigMask = false(size(delta));
    end
    sigMask = logical(sigMask(:));

    % spacing
    if ~isfield(nCh, sessName), error('nCh.%s not defined.', sessName); end
    if nCh.(sessName) == 24
        dz = spacing_um.ch24;
    elseif nCh.(sessName) == 32
        dz = spacing_um.ch32;
    else
        error('nCh.%s must be 24 or 32.', sessName);
    end

    % granular ref
    if ~isfield(granCh, sessName), error('granCh.%s not defined.', sessName); end
    g = granCh.(sessName);
    gCenter = mean(g);

    depth_um = (elec - gCenter) * dz;

    gMin = (min(g) - gCenter) * dz;
    gMax = (max(g) - gCenter) * dz;

    % layer assignment
    layerIdx = nan(size(depth_um));
    layerIdx(depth_um < gMin) = 1;                      % Infra (deep)
    layerIdx(depth_um >= gMin & depth_um <= gMax) = 2;  % Gran
    layerIdx(depth_um > gMax) = 3;                      % Supra (superficial)

    ok = isfinite(delta) & isfinite(depth_um) & isfinite(layerIdx);
    delta    = delta(ok);
    depth_um = depth_um(ok);
    layerIdx = layerIdx(ok);
    sigMask  = sigMask(ok);
end

function plot_layered_depth_lowess(S, titleStr, xlim_fixed)

    darkGray = [0.35 0.35 0.35];

    delta = S.delta;
    depth = S.depth_um;
    sig   = logical(S.sigMask(:));
    gMin  = S.gMin;
    gMax  = S.gMax;

    if isempty(delta)
        axis off; title([titleStr ' (no data)'],'FontWeight','bold'); return;
    end

    % x/y limits
    xlim(xlim_fixed);
    yl = [min(depth) max(depth)];
    if diff(yl)==0, yl = yl + [-100 100]; end
    pady = 0.08*range(yl);
    yl = [yl(1)-pady yl(2)+pady];
    ylim(yl);

    xl = xlim_fixed;

    % shaded layers (no legend)
    p1 = patch([xl(1) xl(2) xl(2) xl(1)], [yl(1) yl(1) gMin gMin], [1.0 0.92 0.92], 'EdgeColor','none', 'FaceAlpha',0.25);
    p2 = patch([xl(1) xl(2) xl(2) xl(1)], [gMin gMin gMax gMax], [0.92 0.92 1.0], 'EdgeColor','none', 'FaceAlpha',0.25);
    p3 = patch([xl(1) xl(2) xl(2) xl(1)], [gMax gMax yl(2) yl(2)], [0.92 1.0 0.92], 'EdgeColor','none', 'FaceAlpha',0.25);
    set([p1 p2 p3], 'HandleVisibility','off');

    xline(0,'k:','HandleVisibility','off');
    yline(0,'k-','LineWidth',1,'HandleVisibility','off');

    % scatter: nonsig then sig
    nonsig = ~sig;
    h_ns  = scatter(delta(nonsig), depth(nonsig), 36, darkGray, 'filled', 'MarkerFaceAlpha',0.70, 'DisplayName','n.s.');
    h_sig = scatter(delta(sig),    depth(sig),    44, 'r',      'filled', 'MarkerFaceAlpha',0.90, 'DisplayName','p < 0.05');

    % LOWESS of delta(depth) drawn as x(delta_smooth) vs y(depth)
    h_lo = gobjects(1);
    if numel(delta) >= 3
        d_s = smooth(depth, delta, 0.35, 'lowess');
        [ys, ord] = sort(depth);
        h_lo = plot(d_s(ord), ys, 'k-', 'LineWidth', 2.0, 'DisplayName','LOWESS');
    end

    xlabel('\Delta = mean\_dFR\_After - mean\_dFR\_Before','FontWeight','bold');
    ylabel('Depth rel. granular center (\mum)  (deep \rightarrow superficial)','FontWeight','bold');
    title(titleStr,'FontWeight','bold');
    set(gca,'YDir','normal');

    % minimal legend
    legH = [h_sig; h_ns];
    legN = {'p < 0.05','n.s.'};
    if isgraphics(h_lo), legH(end+1,1) = h_lo; legN{end+1} = 'LOWESS'; end %#ok<AGROW>
    legend(legH, legN, 'Location','best', 'Box','off');
    set(gca,'FontWeight','bold','FontSize',18,'Layer','top');

end

function plot_layer_summary_bootstrap(S, titleStr, xlim_fixed, bootB, alphaSig)
    % Reliability: 95% bootstrap CI of layer mean, plus %sig and n

    darkGray = [0.35 0.35 0.35];

    delta = S.delta;
    layer = S.layerIdx;
    sig   = logical(S.sigMask(:));

    if isempty(delta)
        axis off; title([titleStr ' (no data)'],'FontWeight','bold'); return;
    end

    layerNames = {'Infra','Gran','Supra'};
    y0 = [1 2 3];

    xlim(xlim_fixed);
    ylim([0.5 3.5]);
    yticks(y0);
    yticklabels(layerNames);
    xline(0,'k:','HandleVisibility','off');

    % legend handles (only once per panel)
    h_ns = gobjects(1); h_sig = gobjects(1); h_ci = gobjects(1);

    for j = 1:3
        idx = (layer == j);
        Dj  = delta(idx);
        Sj  = sig(idx);

        if isempty(Dj), continue; end

        % jittered points
        yj = y0(j) + 0.12*(rand(numel(Dj),1)-0.5);

        % plot nonsig then sig
        idx_ns = ~Sj;
        if any(idx_ns)
            h_ns = scatter(Dj(idx_ns), yj(idx_ns), 40, darkGray, 'filled', ...
                           'MarkerFaceAlpha',0.75, 'DisplayName','n.s.');
        end
        if any(Sj)
            h_sig = scatter(Dj(Sj), yj(Sj), 52, 'r', 'filled', ...
                            'MarkerFaceAlpha',0.90, 'DisplayName',sprintf('p < %.2g', alphaSig));
        end

        % bootstrap 95% CI for mean
        m = mean(Dj);
        ci = boot_ci_mean(Dj, bootB, 0.05);

        % draw CI and mean
        h_ci = plot([ci(1) ci(2)], [y0(j) y0(j)], 'k-', 'LineWidth', 3, 'DisplayName','Mean ± 95% CI');
        plot(m, y0(j), 'ko', 'MarkerFaceColor','w', 'MarkerSize',8, 'LineWidth',1.8, 'HandleVisibility','off');

        % annotate reliability-ish info: n and %sig
        pctSig = 100 * mean(Sj);
        text(xlim_fixed(1) + 0.02*range(xlim_fixed), y0(j), ...
            sprintf('n=%d, %%sig=%.1f', numel(Dj), pctSig), ...
            'FontWeight','bold', 'VerticalAlignment','middle');
        
    end

    xlabel('\Delta = mean\_dFR\_After - mean\_dFR\_Before','FontWeight','bold');
    title([titleStr ' (bootstrap 95% CI)'], 'FontWeight','bold');

    % minimal legend
    legH = [];
    legN = {};

    if isgraphics(h_sig), legH(end+1,1) = h_sig; legN{end+1} = sprintf('p < %.2g', alphaSig); end %#ok<AGROW>
    if isgraphics(h_ns),  legH(end+1,1) = h_ns;  legN{end+1} = 'n.s.'; end %#ok<AGROW>
    if isgraphics(h_ci),  legH(end+1,1) = h_ci;  legN{end+1} = 'Mean ± 95% CI'; end %#ok<AGROW>

    legend(legH, legN, 'Location','best', 'Box','off');
    set(gca,'FontWeight','bold','FontSize',18,'Layer','top');

end

function ci = boot_ci_mean(x, B, alpha)
    % Percentile bootstrap CI for mean
    x = x(:);
    x = x(isfinite(x));
    n = numel(x);
    if n == 0
        ci = [NaN NaN]; return;
    end
    if n == 1
        ci = [x x]; return;
    end
    bootm = zeros(B,1);
    for b = 1:B
        idx = randi(n, n, 1);
        bootm(b) = mean(x(idx));
    end
    lo = prctile(bootm, 100*(alpha/2));
    hi = prctile(bootm, 100*(1-alpha/2));
    ci = [lo hi];
end


