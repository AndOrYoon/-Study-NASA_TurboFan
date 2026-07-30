@echo off
REM H7 Loss Function Optimization - Sequential Phase Runner
REM Run this from the command line:
REM   C:\BMAD_PY313\Data_Analysis\Code\H7_loss_function\run_h7_phases.bat

SET PYTHON=C:\BMAD_PY313\NASA_TurboFan\Scripts\python.exe
SET CODEDIR=C:\BMAD_PY313\Data_Analysis\Code\H7_loss_function

echo ============================================================
echo H7 Loss Function Optimization Pipeline
echo ============================================================

echo.
echo [Phase 0] Pipeline Check
%PYTHON% %CODEDIR%\00_pipeline_check.py
IF ERRORLEVEL 1 (echo FAILED - aborting & exit /b 1)

echo.
echo [Phase 1] Linear Screening (80 runs)
%PYTHON% %CODEDIR%\04_phase1_linear_screening.py
IF ERRORLEVEL 1 echo WARNING: Phase 1 had errors

echo.
echo [Phase 2a] LSTM Pilot (25 runs)
%PYTHON% %CODEDIR%\05_phase2a_lstm_pilot.py
IF ERRORLEVEL 1 echo WARNING: Phase 2a had errors

echo.
echo [Phase 2b] Full LSTM Experiment (560 runs - takes 2-4 hours on RTX 5080)
%PYTHON% %CODEDIR%\05_phase2b_lstm_full.py
IF ERRORLEVEL 1 echo WARNING: Phase 2b had errors

echo.
echo [Phase 3a] L5 Lambda Grid (150 runs)
%PYTHON% %CODEDIR%\06_phase3a_lambda_grid.py
IF ERRORLEVEL 1 echo WARNING: Phase 3a had errors

echo.
echo [Phase 3c] Pinball Tau Sweep (100 runs)
%PYTHON% %CODEDIR%\06_phase3c_pinball_tau.py
IF ERRORLEVEL 1 echo WARNING: Phase 3c had errors

echo.
echo [Phase 3d] HubA Grid (45 runs)
%PYTHON% %CODEDIR%\06_phase3d_huba_grid.py
IF ERRORLEVEL 1 echo WARNING: Phase 3d had errors

echo.
echo [Phase 7] Statistical Evaluation
%PYTHON% %CODEDIR%\07_evaluation.py
IF ERRORLEVEL 1 echo WARNING: Evaluation had errors

echo.
echo [Phase 8] Visualization
%PYTHON% %CODEDIR%\08_visualization.py
IF ERRORLEVEL 1 echo WARNING: Visualization had errors

echo.
echo ============================================================
echo H7 Pipeline Complete
echo Results: C:\BMAD_PY313\Data_Analysis\Results\H7_loss_function\
echo ============================================================
