classdef RainfallPredictorSVRApp < matlab.apps.AppBase

    % Properties
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        LoadDataButton       matlab.ui.control.Button
        PredictButton        matlab.ui.control.Button
        UITable              matlab.ui.control.Table
        ResultsTextArea      matlab.ui.control.TextArea
        MoreInfoLabel        matlab.ui.control.Label 
    end

    properties (Access = private)
        mdl_svr              % SVR model
        ps                   % Normalization settings (mapminmax)
        rawData              % Loaded raw data table
    end

    % Startup function
    methods (Access = private)
        function startupFcn(app)
            if exist('modelGUISVR.mat', 'file')
                s = load('modelGUISVR.mat');
                app.mdl_svr = s.mdl_svr;
                app.ps = s.ps;
            else
                uialert(app.UIFigure, 'ไม่พบไฟล์ modelGUISVR.mat', 'ข้อผิดพลาด');
            end
        end
    end

    % Callbacks
    methods (Access = private)

        % Load Data Button pushed
        function LoadDataButtonPushed(app, event)
            [file, path] = uigetfile('*.xlsx');
            if isequal(file,0)
                return;
            end
            data = readtable(fullfile(path, file));
            data = rmmissing(data);
            app.rawData = data;
            % แสดงข้อมูล 10 แถวแรกในตาราง
            app.UITable.Data = data(1:min(10,height(data)), :);
            app.ResultsTextArea.Value = {'โหลดข้อมูลสำเร็จ'};
        end

        % Predict Button pushed
        function PredictButtonPushed(app, event)
            if isempty(app.rawData)
                uialert(app.UIFigure, 'กรุณาโหลดข้อมูลก่อน', 'คำเตือน');
                return;
            end

            data = app.rawData;

            % กำหนดฟีเจอร์ที่ใช้
            features = {'MaxAirPressure','MinAirPressure','AvgAirPressure8Time', ...
                        'MaxTemp','MinTemp','AvgTemp','Evaporation', ...
                        'MaxHumidity','MinHumidity','AvgHumidity'};

            % เตรียมข้อมูล X และ y
            X_raw = data{:, features};
            y_raw = data.Rainfall;

            % เลื่อน y เพื่อพยากรณ์วันถัดไป (shift=1)
            shift = 1;
            y_shifted = [y_raw(shift:end); NaN(shift,1)];
            valid_idx = ~isnan(y_shifted);

            X = X_raw(valid_idx, :);
            y_true = y_shifted(valid_idx);

            % วันที่ (ถ้ามี)
            if ismember('Date', data.Properties.VariableNames)
                dates = data{valid_idx, 'Date'};
            else
                dates = repmat(NaT, sum(valid_idx), 1);
            end

            % Normalize features ด้วย mapminmax ที่โหลดมา
            X_scaled = mapminmax('apply', X', app.ps)';

            % Predict
            y_pred = predict(app.mdl_svr, X_scaled);
            y_pred(y_pred < 0) = 0;

            % ฟังก์ชันจัดกลุ่มฝน
            classifyRain = @(val) ...
                "ไม่มีฝน"*(val < 0.1) + ...
                "น้อย"*(val >= 0.1 & val <= 10) + ...
                "ปานกลาง"*(val > 10 & val <= 35) + ...
                "หนัก"*(val > 35 & val <= 90) + ...
                "หนักมาก"*(val > 90);

            % ใช้ if-elseif แทนการคูณ string ไม่ได้ใน MATLAB
            rain_class_pred = strings(length(y_pred),1);
            rain_class_true = strings(length(y_true),1);
            for i = 1:length(y_pred)
                val_pred = y_pred(i);
                if val_pred < 0.1
                    rain_class_pred(i) = "ฝนวันไม่ได้ ⬛";
                elseif val_pred <= 10
                    rain_class_pred(i) = "ฝนตกน้อยน้อย 🟦";
                elseif val_pred <= 35
                    rain_class_pred(i) = "ฝนตกปานกลาง 🟩";
                elseif val_pred <= 90
                    rain_class_pred(i) = "ฝนตกหนัก 🟨";
                else
                    rain_class_pred(i) = "ฝนตกหนักมาก 🟥";
                end

                val_true = y_true(i);
                if val_true < 0.1
                    rain_class_true(i) = "ฝนวันไม่ได้ ⬛";
                elseif val_true <= 10
                    rain_class_true(i) = "ฝนตกน้อย 🟦";
                elseif val_true <= 35
                    rain_class_true(i) = "ฝนตกปานกลาง 🟩";
                elseif val_true <= 90
                    rain_class_true(i) = "ฝนตกหนัก 🟨";
                else
                    rain_class_true(i) = "ฝนตกหนักมาก 🟥";
                end
            end

            % สร้างตารางผลลัพธ์
            result_table = table(dates, y_true, y_pred, rain_class_true, rain_class_pred, ...
                'VariableNames', {'วันที่', 'ค่าจริง_mm', 'ค่าทำนาย_mm', 'กลุ่มค่าจริง', 'กลุ่มค่าทำนาย'});
            app.UITable.Data = result_table;


            app.ResultsTextArea.Value = {'พยากรณ์เสร็จสิ้น ✅ '};
        end

    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = RainfallPredictorSVRApp
            createComponents(app)
            runStartupFcn(app, @(app)startupFcn(app))
        end

        % Delete UIFigure when app is deleted
        function delete(app)
            delete(app.UIFigure)
        end
    end

    % Component initialization
    methods (Access = private)

        function createComponents(app)

            % Create UIFigure
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 700 450];
            app.UIFigure.Name = 'Rainfall Predictor - SVR';
        
            % เพิ่มตรงนี้เพื่อเปิดหน้าต่างแบบเต็มจอ
             app.UIFigure.WindowState = 'maximized';

            % Load Data Button
            app.LoadDataButton = uibutton(app.UIFigure, 'push');
            app.LoadDataButton.Position = [20 390 100 30];
            app.LoadDataButton.Text = 'โหลดข้อมูล';
            app.LoadDataButton.ButtonPushedFcn = createCallbackFcn(app, @LoadDataButtonPushed, true);
            app.LoadDataButton.FontSize = 18;  % ตั้งขนาดฟอนต์

            % Predict Button
            app.PredictButton = uibutton(app.UIFigure, 'push');
            app.PredictButton.Position = [140 390 100 30];
            app.PredictButton.Text = 'พยากรณ์';
            app.PredictButton.ButtonPushedFcn = createCallbackFcn(app, @PredictButtonPushed, true);
            app.PredictButton.FontSize = 18;
            

            % UITable for results
            app.UITable = uitable(app.UIFigure);
            app.UITable.Position = [20 150 660 230];
            app.UITable.FontSize = 18;

            % Results Text Area
            app.ResultsTextArea = uitextarea(app.UIFigure);
            app.ResultsTextArea.Position = [20 100 660 30];
            app.ResultsTextArea.Editable = false;
            app.ResultsTextArea.Value = {'รอผลลัพธ์พยากรณ์...'};
            app.ResultsTextArea.FontSize = 40;
            app.ResultsTextArea.HorizontalAlignment = 'center';

            app.MoreInfoLabel = uilabel(app.UIFigure);
            app.MoreInfoLabel.Position = [600 390 100 30]; % ปรับตามต้องการ
            app.MoreInfoLabel.Text = 'SVR Model Prediction Rainfall';
            app.MoreInfoLabel.FontSize = 25;

            app.MoreInfoLabel = uilabel(app.UIFigure);
            app.MoreInfoLabel.Position = [20 35 660 30]; % ปรับตามต้องการ
            app.MoreInfoLabel.Text = 'กลุ่มฝน: ⬛ ฝนวัดไม่ได้ (<0.1 mm)  🟦 ฝนตกน้อย (0.1-10 mm) 🟩 ฝนตกปานกลาง (10.1-35 mm) 🟨 ฝนตกหนัก (35.1-90 mm) 🟥 ฝนตกหนักมาก (>90 mm)';
            app.MoreInfoLabel.FontSize = 18;
            app.MoreInfoLabel.HorizontalAlignment = 'center';

            app.UIFigure.Visible = 'on';
        end

    end
end

