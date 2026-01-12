clear;
 rng(42);  %  reproducibility

% 1. Load Data
filename = 'data_rainfall.xlsx';
data = readtable(filename);

if ismember('Date', data.Properties.VariableNames)
    data.Date = [];
end


% 2. Feature Selection
% feature_cols = {'AvgHumidity', 'AvgAirPressure8Time', 'AvgTemp'};
feature_cols = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time',...
                'MaxTemp','MinTemp','AvgTemp','Evaporation',...
                'MaxHumidity','MinHumidity','AvgHumidity'};
% feature_cols = {
%     'AvgHumidity','MinHumidity','MaxHumidity','AvgAirPressure8Time','MaxAirPressure','Evaporation'};

X = data{:, feature_cols};
y = data.Rainfall;
% y = data.RainfallNomar;

% 3. Shift y (Predict rainfall of next day)
shift = 1;  % พยากรณ์ 1 วันข้างหน้า
y_shifted = [y(shift:end); NaN(shift,1)];

% 4. Remove missing data
valid_idx = ~isnan(y_shifted);
X = X(valid_idx, :);
y_shifted = y_shifted(valid_idx);

% 5. Create Sliding Window
timeStep = 7;
X_seq = {}; 
y_seq = [];

for i = timeStep+1 : size(X, 1)
    X_seq{end+1, 1} = X(i-timeStep:i-1, :)';
    y_seq(end+1, 1) = y_shifted(i);
end

% 6. Split Train/Test
train_ratio = 0.8;
numTrain = floor(train_ratio * length(y_seq));

X_train = X_seq(1:numTrain);
y_train = y_seq(1:numTrain);
X_test = X_seq(numTrain+1:end);
y_test = y_seq(numTrain+1:end);

% 7. Normalize X (Min-Max Scaling)
XtrainMat = cat(3, X_train{:});
XtestMat = cat(3, X_test{:});

X_min = min(XtrainMat, [], [2 3]);
X_max = max(XtrainMat, [], [2 3]);

for i = 1:length(X_train)
    X_train{i} = (X_train{i} - X_min) ./ (X_max - X_min + eps);
end

for i = 1:length(X_test)
    X_test{i} = (X_test{i} - X_min) ./ (X_max - X_min + eps);
end


% 8. Build LSTM Regression Model
%  load('H32run5.mat', 'net')
numFeatures = size(X_train{1}, 1);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50 , 'OutputMode', 'last') 
    fullyConnectedLayer(1)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs', 1500, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'Verbose', 1);


% 9. Train
% net = trainNetwork(X_train , y_train, net.Layers , options);
 net = trainNetwork(X_train , y_train, layers , options);
 % save('modelGUILSTM', 'net');

% 10. Predict
y_train_pred = predict(net, X_train);
y_test_pred = predict(net, X_test);

y_train_pred(y_train_pred < 0) = 0;
y_test_pred(y_test_pred < 0) = 0;

% 11. Evaluation (Regression)
MAE_train = mean(abs(y_train - y_train_pred));
RMSE_train = sqrt(mean((y_train - y_train_pred).^2));
R2_train = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);

MAE_test = mean(abs(y_test - y_test_pred));
RMSE_test = sqrt(mean((y_test - y_test_pred).^2));
R2_test = 1 - sum((y_test - y_test_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\nTraining Set: MAE : %.4f, RMSE : %.4f, R2 : %.4f\n', MAE_train, RMSE_train, R2_train);
fprintf('Testing Set: MAE : %.4f, RMSE : %.4f, R2 : %.4f\n', MAE_test, RMSE_test, R2_test);

figure;
subplot(2,1,1);
plot(y_train, 'b', 'LineWidth', 1.5); hold on;
plot(y_train_pred, 'r--', 'LineWidth', 1.5);
title('Training Set: Actual vs Predicted Rainfall');
legend('Actual', 'Predicted');
xlabel('Samples');
ylabel('Rainfall (mm)');
grid on;

subplot(2,1,2);
plot(y_test, 'b', 'LineWidth', 1.5); hold on;
plot(y_test_pred, 'r--', 'LineWidth', 1.5);
title('Test Set: Actual vs Predicted Rainfall');
legend('Actual', 'Predicted');
xlabel('Samples');
ylabel('Rainfall (mm)');
grid on;

% 11.5 Confusion Matrix + Class Metrics

% ฟังก์ชันแบ่งระดับฝน
categorizeRainfall = @(x) (x < 0.1) * 1 + ...
                           (x >= 0.1 & x <= 10) * 2 + ...
                           (x > 10 & x <= 35) * 3 + ...
                           (x > 35 & x <= 90) * 4 + ...
                           (x > 90) * 5;

% แปลงค่าจริงและค่าพยากรณ์เป็นกลุ่ม
actual_classes = arrayfun(categorizeRainfall, y_test);
pred_classes = arrayfun(categorizeRainfall, y_test_pred);

% สร้าง Confusion Matrix
C = confusionmat(actual_classes, pred_classes);

% ชื่อกลุ่มฝน
labels = {'ฝนวัดจำนวนไม่ได้', 'ฝนตกน้อย', 'ฝนตกปานกลาง', 'ฝนตกหนัก', 'ฝนตกหนักมาก'};

% แสดง Confusion Matrix
fprintf('\nConfusion Matrix:\n');
fprintf('%20s', '');
for i = 1:size(C,1)
    fprintf('%15s', labels{i});
end
fprintf('\n');
for i = 1:size(C,1)
    fprintf('%20s', labels{i});
    for j = 1:size(C,2)
        fprintf('%15d', C(i,j));
    end
    fprintf('\n');
end

% คำนวณ Metrics
num_classes = size(C, 1);
accuracy = sum(diag(C)) / sum(C(:));
precision = zeros(num_classes,1);
recall = zeros(num_classes,1);
f1_score = zeros(num_classes,1);

for i = 1:num_classes
    TP = C(i,i);
    FP = sum(C(:,i)) - TP;
    FN = sum(C(i,:)) - TP;
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% MAE เฉพาะตอนที่มีฝน
MAE_rain = mean(abs(y_test(y_test > 0) - y_test_pred(y_test > 0)));

% แสดงผล Metrics
fprintf('\nMAE (เฉพาะฝนตก): %.4f mm\n', MAE_rain);

fprintf('\nModel Performance Metrics (Per Class):\n');
for i = 1:num_classes
    fprintf('%-20s | Precision: %.4f | Recall: %.4f | F1-Score: %.4f\n', labels{i}, precision(i), recall(i), f1_score(i));
end

fprintf('\nOverall Accuracy: %.2f%%\n', accuracy * 100);