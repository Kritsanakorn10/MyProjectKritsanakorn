clc; clear; close all;
rng(42);  % reproducibility (จะไม่มีผลกับการแบ่งแบบไม่สุ่ม)

%% Load data
filename = 'data_rainfall.xlsx';
data = readtable(filename);
if ismember('Date', data.Properties.VariableNames)
    data.Date = [];
end

data = rmmissing(data);  % Remove NaN rows

%% Feature selection
feature_cols = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time',...
                'MaxTemp','MinTemp','AvgTemp','Evaporation',...
                'MaxHumidity','MinHumidity','AvgHumidity'};

X_raw = data{:, feature_cols};
y_raw = data.Rainfall;

%% Shift y for next-day prediction
shift = 1;
y_shifted = [y_raw(shift:end); NaN(shift,1)];

% Remove rows with NaN in shifted y
valid_idx = ~isnan(y_shifted);
X_all = X_raw(valid_idx, :);
y_all = y_shifted(valid_idx);

%% K-Fold Split แบบไม่สุ่ม (Sequential Split)
k = 5;
N = length(y_all);

fold_sizes = floor(N / k) * ones(k,1);
remainder = mod(N, k);
fold_sizes(1:remainder) = fold_sizes(1:remainder) + 1;

indices = zeros(N,1);
start_idx = 1;
for i = 1:k
    end_idx = start_idx + fold_sizes(i) - 1;
    indices(start_idx:end_idx) = i;
    start_idx = end_idx + 1;
end

% Prepare for metric storage
all_confusion = zeros(5,5);
Accs = zeros(k,1);
Precs = zeros(k,1);
Recs = zeros(k,1);
F1s = zeros(k,1);

% Rainfall classification function
categorizeRainfall = @(x) (x < 0.1) * 1 + ...
                          (x >= 0.1 & x <= 10) * 2 + ...
                          (x > 10 & x <= 35) * 3 + ...
                          (x > 35 & x <= 90) * 4 + ...
                          (x > 90) * 5;

%% K-Fold Loop
for fold = 1:k
    fprintf('\nFold %d/%d\n', fold, k);

    % Train/test split แบบไม่สุ่ม
    train_idx = (indices ~= fold);
    test_idx = (indices == fold);

    X_train = X_all(train_idx, :);
    y_train = y_all(train_idx);
    X_test  = X_all(test_idx, :);
    y_test  = y_all(test_idx);

    % Normalize
    [X_train_scaled, ps] = mapminmax(X_train');
    X_train_scaled = X_train_scaled';
    X_test_scaled = mapminmax('apply', X_test', ps)';

    % Train SVR
    mdl = fitrsvm(X_train_scaled, y_train, ...
                  'KernelFunction', 'rbf', ...
                  'BoxConstraint', 180000, ...
                  'Epsilon', 0.05);

    % Predict
    y_pred = predict(mdl, X_test_scaled);
    y_pred(y_pred < 0) = 0;

    % Classification
    actual_classes = arrayfun(categorizeRainfall, y_test);
    pred_classes   = arrayfun(categorizeRainfall, y_pred);
    C = confusionmat(actual_classes, pred_classes, 'Order', 1:5);
    all_confusion = all_confusion + C;

    % Per-fold metrics
    num_classes = 5;
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;
    TN = sum(C(:)) - (TP + FP + FN);

    Precision = mean(TP ./ (TP + FP + eps));
    Recall    = mean(TP ./ (TP + FN + eps));
    F1        = mean(2 * (Precision * Recall) ./ (Precision + Recall + eps));
    Accuracy  = mean((TP + TN) ./ (TP + TN + FP + FN + eps));

    % Store
    Accs(fold) = Accuracy;
    Precs(fold) = Precision;
    Recs(fold) = Recall;
    F1s(fold) = F1;

    % Show per fold
    fprintf('Accuracy : %.4f\n', Accuracy);
    fprintf('Precision: %.4f\n', Precision);
    fprintf('Recall   : %.4f\n', Recall);
    fprintf('F1-Score : %.4f\n', F1);
end

%% Show Confusion Matrix
labels = {'ฝนวัดไม่ได้', 'ฝนตกน้อย', 'ฝนปานกลาง', 'ฝนหนัก', 'ฝนหนักมาก'};
fprintf('\nAggregated Confusion Matrix:\n');
fprintf('%20s', '');
for i = 1:5
    fprintf('%15s', labels{i});
end
fprintf('\n');
for i = 1:5
    fprintf('%20s', labels{i});
    for j = 1:5
        fprintf('%15d', all_confusion(i,j));
    end
    fprintf('\n');
end

%% Summary of Metrics
fprintf('\nSummary (Average over %d Folds):\n', k);
fprintf('Accuracy : %.4f ± %.4f\n', mean(Accs), std(Accs));
fprintf('Precision: %.4f ± %.4f\n', mean(Precs), std(Precs));
fprintf('Recall   : %.4f ± %.4f\n', mean(Recs), std(Recs));
fprintf('F1-Score : %.4f ± %.4f\n', mean(F1s), std(F1s));
