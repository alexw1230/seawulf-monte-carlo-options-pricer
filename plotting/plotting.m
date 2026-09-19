clear; clc; close all;

CSV_PATH = '../../combined_sweep.csv';

% (S0=100, K=100, r=0.05, sigma=0.20, T=1)
BS_CALL = 10.450584;
BS_PUT  = 5.573526;

data = csvread(CSV_PATH, 1, 0);   % skip header row
P_col         = data(:,1);
N_col         = data(:,2);
call_col      = data(:,3);
call_se_col   = data(:,4);
total_s_col   = data(:,9);

P_values = unique(P_col);   % sorted ascending
N_values = unique(N_col);   % sorted ascending

% mean_total(i,j) = mean total_s for P_values(i), N_values(j)
mean_total = zeros(length(P_values), length(N_values));
for i = 1:length(P_values)
    for j = 1:length(N_values)
        mask = (P_col == P_values(i)) & (N_col == N_values(j));
        mean_total(i,j) = mean(total_s_col(mask));
    end
end

%% ---- Plot 1: runtime vs N, one line per P ----
figure('Position', [100 100 800 600]);
hold on;
colors_p = lines(length(P_values));
for i = 1:length(P_values)
    plot(N_values, mean_total(i,:), '-o', 'Color', colors_p(i,:), ...
        'DisplayName', sprintf('P=%d', P_values(i)), 'LineWidth', 1.5);
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('N (simulations)');
ylabel('Total time (sec)');
title('Total runtime vs N, by P');
legend('Location', 'eastoutside');
grid on;
print(gcf, 'runtime_vs_N.png', '-dpng', '-r300');

%% ---- Speedup and efficiency, relative to the P=1 row ----
p1_idx = find(P_values == 1);
speedup    = bsxfun(@rdivide, mean_total(p1_idx, :), mean_total);   % (P x N)
efficiency = bsxfun(@rdivide, speedup, P_values);                    % divide each row by its P

%% ---- Plot 2: speedup vs P, one line per N, plus ideal reference ----
figure('Position', [100 100 800 600]);
hold on;
colors_n = lines(length(N_values));
for j = 1:length(N_values)
    plot(P_values, speedup(:,j), '-o', 'Color', colors_n(j,:), ...
        'DisplayName', sprintf('N=%.0e', N_values(j)), 'LineWidth', 1.5);
end
plot(P_values, P_values, '--', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Ideal', 'LineWidth', 1.5);
xlabel('P (cores)');
ylabel('Speedup T(1)/T(P)');
title('Speedup vs P, by N');
legend('Location', 'eastoutside');
grid on;
print(gcf, 'speedup_vs_P.png', '-dpng', '-r300');

%% ---- Plot 3: efficiency vs P, one line per N ----
figure('Position', [100 100 800 600]);
hold on;
for j = 1:length(N_values)
    plot(P_values, efficiency(:,j), '-o', 'Color', colors_n(j,:), ...
        'DisplayName', sprintf('N=%.0e', N_values(j)), 'LineWidth', 1.5);
end
plot([min(P_values) max(P_values)], [1 1], '--', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Ideal (1.0)', 'LineWidth', 1.5);
ylim([0 1.05]);
xlabel('P (cores)');
ylabel('Efficiency (speedup / P)');
title('Parallel efficiency vs P, by N');
legend('Location', 'eastoutside');
grid on;
print(gcf, 'efficiency_vs_P.png', '-dpng', '-r300');

%% ---- Plot 4 (bonus): call price vs N with 95% CI, one fixed P ----
% Price/SE are, in principle, independent of how the work was split across
% ranks -- but different P values use different SeedSequence.spawn(P)
% child streams, so they are NOT bit-identical across P at the same N.
% Picking one P keeps this an apples-to-apples convergence plot.
PLOT_P = max(P_values);
mask_p = (P_col == PLOT_P);
Np = N_col(mask_p);
callp = call_col(mask_p);
call_sep = call_se_col(mask_p);

[Np_sorted, sort_idx] = sort(Np);
callp_sorted = callp(sort_idx);
call_sep_sorted = call_sep(sort_idx);

% collapse to unique N (repeats give identical price/SE by design -- same
% seed every repeat -- so just take one row per N here)
[N_unique, ia] = unique(Np_sorted);
call_unique = callp_sorted(ia);
call_se_unique = call_sep_sorted(ia);

figure('Position', [100 100 800 600]);
hold on;
errorbar(N_unique, call_unique, 1.96*call_se_unique, '-o', ...
    'LineWidth', 1.5, 'DisplayName', sprintf('MC estimate (P=%d), 95%% CI', PLOT_P));
plot([min(N_unique) max(N_unique)], [BS_CALL BS_CALL], '--', 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 1.5, 'DisplayName', 'Black-Scholes');
set(gca, 'XScale', 'log');
xlabel('N (simulations)');
ylabel('Call price ($)');
title('Monte Carlo call price convergence vs N');
legend('Location', 'best');
grid on;
print(gcf, 'price_convergence.png', '-dpng', '-r300');

fprintf('Saved: runtime_vs_N.png, speedup_vs_P.png, efficiency_vs_P.png, price_convergence.png\n');