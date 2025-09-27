@echo off
REM Runs the entire pipeline using Python module syntax (-m).
REM The module paths are based on the exact project structure.
REM RUN THIS SCRIPT FROM THE PROJECT'S ROOT FOLDER.

setlocal

echo ===== STEP 1: TRAINING THE TGN MODEL =====
python -m src.training.train_tgn
if %errorlevel% neq 0 ( echo [ERROR] Training failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 2: MODEL CALIBRATION =====
python -m src.postprocess.calibrate
if %errorlevel% neq 0 ( echo [ERROR] Calibration failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 3: SCORING (NOISY DATASET) =====
python -m src.inference.score_events --input input/edge_events.csv --output outputs/anomalies.jsonl
if %errorlevel% neq 0 ( echo [ERROR] Noisy scoring failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 4: SCORING (CLEAN DATASET) =====
python -m src.inference.score_events --input input/edge_events_clean.csv --output outputs/anomalies_clean.jsonl
if %errorlevel% neq 0 ( echo [ERROR] Clean scoring failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 5: RUNNING PRELIMINARY ANALYSES =====
python -m src.analysis.analysis
python -m src.analysis.compare_clean_noisy
python -m src.analysis.summary_anomalies
python -m src.postprocess.analyze_graph
if %errorlevel% neq 0 ( echo [ERROR] Preliminary analysis failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 6: PERFORMANCE EVALUATION =====
python -m src.training.evaluate
if %errorlevel% neq 0 ( echo [ERROR] Evaluation failed. & exit /b %errorlevel% )

echo.
echo ===== STEP 7: GENERATING FINAL REPORT (CHARTS) =====
python -m src.postprocess.generate_report
if %errorlevel% neq 0 ( echo [ERROR] Report generation failed. & exit /b %errorlevel% )

echo.
echo.
echo PIPELINE SUCCESSFULLY COMPLETED!
echo The final results, including charts, can be found in the 'outputs/' folder.

endlocal

Pause