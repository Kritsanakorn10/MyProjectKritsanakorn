clear; clc;
rng(42);  % reproducibility

%% Load Data
filename = 'data_rainfall.xlsx';
data = readtable(filename);

if ismember('Date', data.Properties.VariableNames)
    data.Date = [];
end

feature_cols = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time',...
                'MaxTemp','MinTemp','AvgTemp','Evaporation',...
                'MaxHumidity','MinHumidity','AvgHumidity'};
X = data{:, feature_cols};
y = data.Rainfall;

%% Shift y (Predict rainfall of next day)
shift = 1;
y_shifted = [y(shift:end); NaN(shift,1)];
valid_idx = ~isnan(y_shifted);
X = X(valid_idx, :);
y_shifted = y_shifted(valid_idx);

%% Create Sliding Window
timeStep = 7;
X_seq = {};
y_seq = [];

for i = timeStep+1 : size(X, 1)
    X_seq{end+1, 1} = X(i-timeStep:i-1, :)';
    y_seq(end+1, 1) = y_shifted(i);
end

%% Walk-Forward Validation Setup
k = 5;
N = length(y_seq);
step_size = floor(N / k);

all_confusion = zeros(5,5);
fold_metrics = struct('Accuracy', [], 'Precision', [], 'Recall', [], 'F1', []);

for fold = 1:k
    fprintf('\nFold %d/%d\n', fold, k);

    train_end = fold * step_size;
    test_start = train_end + 1;
    test_end = min((fold + 1) * step_size, N);

    if test_start > N || test_end <= train_end
        break;
    end

    X_train = X_seq(1:train_end);
    y_train = y_seq(1:train_end);
    X_test  = X_seq(test_start:test_end);
    y_test  = y_seq(test_start:test_end);

    %% Normalize (Min-Max Scaling)
    XtrainMat = cat(3, X_train{:});
    X_min = min(XtrainMat, [], [2 3]);
    X_max = max(XtrainMat, [], [2 3]);

    for i = 1:length(X_train)
        X_train{i} = (X_train{i} - X_min) ./ (X_max - X_min + eps);
    end
    for i = 1:length(X_test)
        X_test{i} = (X_test{i} - X_min) ./ (X_max - X_min + eps);
    end

    %% LSTM Model
    numFeatures = size(X_train{1}, 1);
    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(50 , 'OutputMode', 'last')
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 1500, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 0.01, ...
        'Verbose', 0);

    net = trainNetwork(X_train, y_train, layers, options);

    %% Predict
    y_test_pred = predict(net, X_test);
    y_test_pred(y_test_pred < 0) = 0;

    %% Classification Mapping
    categorizeRainfall = @(x) (x < 0.1) * 1 + ...
                              (x >= 0.1 & x <= 10) * 2 + ...
                              (x > 10 & x <= 35) * 3 + ...
                              (x > 35 & x <= 90) * 4 + ...
                              (x > 90) * 5;

    actual_classes = arrayfun(categorizeRainfall, y_test);
    pred_classes   = arrayfun(categorizeRainfall, y_test_pred);

    %% Confusion Matrix (Fixed Size 5x5)
    C = confusionmat(actual_classes, pred_classes, 'Order', 1:5);
    all_confusion = all_confusion + C;

    %% Metrics for this fold
    num_classes = size(C,1);
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;
    TN = sum(C(:)) - (TP + FP + FN);

    Precision = mean(TP ./ (TP + FP + eps));
    Recall    = mean(TP ./ (TP + FN + eps));
    F1        = mean(2 * (Precision * Recall) ./ (Precision + Recall + eps));
    Accuracy  = mean((TP + TN) ./ (TP + TN + FP + FN + eps));

    fold_metrics(fold).Accuracy  = Accuracy;
    fold_metrics(fold).Precision = Precision;
    fold_metrics(fold).Recall    = Recall;
    fold_metrics(fold).F1        = F1;

    fprintf('Accuracy : %.4f\n', Accuracy);
    fprintf('Precision: %.4f\n', Precision);
    fprintf('Recall   : %.4f\n', Recall);
    fprintf('F1-Score : %.4f\n', F1);
end

%% Summary Confusion Matrix
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

%% Summary Metrics
Accs  = [fold_metrics.Accuracy];
Precs = [fold_metrics.Precision];
Recs  = [fold_metrics.Recall];
F1s   = [fold_metrics.F1];

fprintf('\nAverage over %d Walk-Forward Folds:\n', k);
fprintf('Accuracy : %.4f ± %.4f\n', mean(Accs), std(Accs));
fprintf('Precision: %.4f ± %.4f\n', mean(Precs), std(Precs));
fprintf('Recall   : %.4f ± %.4f\n', mean(Recs), std(Recs));
fprintf('F1-Score : %.4f ± %.4f\n', mean(F1s), std(F1s));
