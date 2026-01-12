clc; clear; close all;
 rng(42);  %  reproducibility
% โหลดข้อมูล
filename = 'data_rainfall.xlsx';
data = readtable(filename);

% ลบคอลัมน์ Date
if ismember('Date', data.Properties.VariableNames)
    data.Date = [];
end


% ลบแถวที่มี NaN
data = rmmissing(data);


feature_cols = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time',...
                'MaxTemp','MinTemp','AvgTemp','Evaporation',...
                'MaxHumidity','MinHumidity','AvgHumidity'};

% สร้าง X และ y
X_raw = data{:, feature_cols};
y_raw = data.Rainfall;


% Shift y เพื่อพยากรณ์ "วันถัดไป"
shift = 1;
y_shifted = [y_raw(shift:end); NaN(shift, 1)];


% ลบแถวที่ y เป็น NaN
valid_idx = ~isnan(y_shifted);
X = X_raw(valid_idx, :);
y = y_shifted(valid_idx);

% แบ่งข้อมูลเป็น Train/Test
n = size(X,1);
train_size = round(0.8 * n);

X_train = X(1:train_size, :);
y_train = y(1:train_size, :);
X_test  = X(train_size+1:end, :);
y_test  = y(train_size+1:end, :);

% Normalize (Min-Max Scaling)
[X_train_scaled, ps] = mapminmax(X_train');
X_train_scaled = X_train_scaled';
X_test_scaled = mapminmax('apply', X_test', ps)';

% สร้าง SVR Model
mdl_svr = fitrsvm(X_train_scaled, y_train, ...
                  'KernelFunction', 'rbf', ...
                  'BoxConstraint', 180000 , ...
                  'Epsilon', 0.05);

% ทำนายผล
y_train_pred = predict(mdl_svr, X_train_scaled);
y_test_pred  = predict(mdl_svr, X_test_scaled);

% ตัดค่าติดลบ
y_train_pred(y_train_pred < 0) = 0;
y_test_pred(y_test_pred < 0) = 0;

save('modelGUISVR.mat', 'mdl_svr', 'ps');

% ประเมินผล
metrics = @(y_true, y_pred) struct( ...
    'MAE', mean(abs(y_pred - y_true)), ...
    'MSE', mean((y_pred - y_true).^2), ...
    'RMSE', sqrt(mean((y_pred - y_true).^2)), ...
    'R2', 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

results_train = metrics(y_train, y_train_pred);
results_test  = metrics(y_test, y_test_pred);

fprintf('\n Performance:\n');
fprintf('Train - MAE: %.4f, RMSE: %.4f, R²: %.4f\n', ...
    results_train.MAE, results_train.RMSE, results_train.R2);
fprintf('Test  - MAE: %.4f, RMSE: %.4f, R²: %.4f\n', ...
    results_test.MAE, results_test.RMSE, results_test.R2);

% กราฟผลลัพธ์
figure;
subplot(2,1,1);
plot(y_train, 'b', 'LineWidth', 1.5); hold on;
plot(y_train_pred, 'r--', 'LineWidth', 1.5);
legend('Actual Train', 'Predicted Train');
title('Training Set: Actual vs Predicted');
xlabel('Index'); ylabel('Rainfall (mm)'); grid on;

subplot(2,1,2);
plot(y_test, 'b', 'LineWidth', 1.5); hold on;
plot(y_test_pred, 'r--', 'LineWidth', 1.5);
legend('Actual Test', 'Predicted Test');
title('Testing Set: Actual vs Predicted');
xlabel('Index'); ylabel('Rainfall (mm)'); grid on;

sgtitle('SVR: Predict Next-Day Rainfall');

%% Confusion Matrix + Class Metrics

% ฟังก์ชันแบ่งระดับฝน
categorizeRainfall = @(x) (x < 0.1) * 1 + ...
                           (x >= 0.1 & x <= 10) * 2 + ...
                           (x > 10 & x <= 35) * 3 + ...
                           (x > 35 & x <= 90) * 4 + ...
                           (x > 90) * 5;

% แปลงค่าจริงและค่าพยากรณ์เป็นกลุ่ม
actual_classes = arrayfun(categorizeRainfall, y_test);
pred_classes   = arrayfun(categorizeRainfall, y_test_pred);

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
% fprintf('\nMAE (เฉพาะฝนตก): %.4f mm\n', MAE_rain);
fprintf('\nModel Performance Metrics (Per Class):\n');
for i = 1:num_classes
    fprintf('%-20s | Precision: %.4f | Recall: %.4f | F1-Score: %.4f\n', ...
        labels{i}, precision(i), recall(i), f1_score(i));
end

fprintf('\nOverall Accuracy: %.2f%%\n', accuracy * 100);

