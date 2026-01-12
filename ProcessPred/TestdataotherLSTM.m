clear;
 rng(42);  %  reproducibility

% 1. Load Data
filename = 'data_rainfall.xlsx';
data = readtable(filename);

if ismember('Date', data.Properties.VariableNames)
    data.Date = [];
end


% 2. Feature Selection
feature_cols = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time',...
                'MaxTemp','MinTemp','Evaporation','Rainfall',...
                'MaxHumidity','MinHumidity','AvgHumidity'};
% feature_cols = {'MaxTemp', 'MinTemp' , 'AvgTemp' };


X = data{:, feature_cols};
y = data.AvgTemp;


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
    lstmLayer(16 , 'OutputMode', 'last') 
    fullyConnectedLayer(1)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs', 3000, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.1, ...
    'Verbose', 1);


% 9. Train
% net = trainNetwork(X_train , y_train, net.Layers , options);
 net = trainNetwork(X_train , y_train, layers , options);
  % save('modelGUILSTMNewTarget', 'net');

% 10. Predict
y_train_pred = predict(net, X_train);
y_test_pred = predict(net, X_test);

% y_train_pred(y_train_pred < 0) = 0;
% y_test_pred(y_test_pred < 0) = 0;

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
title('Training Set: Actual vs Predicted ');
legend('Actual', 'Predicted');
xlabel('Samples');
ylabel('Target');
grid on;

subplot(2,1,2);
plot(y_test, 'b', 'LineWidth', 1.5); hold on;
plot(y_test_pred, 'r--', 'LineWidth', 1.5);
title('Test Set: Actual vs Predicted ');
legend('Actual', 'Predicted');
xlabel('Samples');
ylabel('Target');
grid on;


% ฟังก์ชันแปลงอุณหภูมิเป็นกลุ่ม class
tempToClass = @(temp) ...
    (temp <= 24.0) * 1 + ...
    (temp > 24.0 & temp <= 30.0) * 2 + ...
    (temp > 30.0) * 3;



y_test_class = arrayfun(tempToClass, y_test);
y_test_pred_class = arrayfun(tempToClass, y_test_pred);

% สร้าง confusion matrix

confMatTest = confusionmat(y_test_class, y_test_pred_class);

% แสดง confusion matrix แบบ text ใน command window
classNames = {'เย็น', 'อบอ้าว', 'ร้อน'};


disp('Confusion Matrix - Testing Set:');
displayConfusionMatrixText(confMatTest, classNames);

% คำนวณ accuracy

accTest = sum(diag(confMatTest)) / sum(confMatTest(:));

fprintf('Testing Accuracy: %.2f%%\n', accTest*100);

% ฟังก์ชันแสดง confusion matrix ในรูปแบบ text
function displayConfusionMatrixText(confMat, classNames)
    n = length(classNames);
    fprintf('%10s', '');
    for j = 1:n
        fprintf('%10s', classNames{j});
    end
    fprintf('\n');
    for i = 1:n
        fprintf('%10s', classNames{i});
        for j = 1:n
            fprintf('%10d', confMat(i,j));
        end
        fprintf('\n');
    end
end
