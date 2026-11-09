@echo off
setlocal
rem Constrained V-LoRA on CMLR. Run this file yourself. It does not start on its own.
rem Visual conv LoRA, rank 4, alpha 8, correlation weight 0.05.
rem No identity loss, no LoRA-MoE, no CVSR-Bili, no Padding.
rem Three learning rates, 7 epochs. Epochs are 0-based: model_0.pth through model_6.pth.
rem Every epoch is saved. After that training finishes, each saved adapter is evaluated.
rem Speakers s2 and s6, adaptation minutes 1/3/5, held-out list with no minute in the name.
rem Eval results for one epoch go to output/test--<train name>_ep<N>_cmlr/log/result_all.txt.
rem
rem lr010  0.010
rem lr015  0.015   paper learning rate
rem lr020  0.020
rem
rem Paper targets at their own 5-epoch schedule, character CER, mean (S2+S6)/2:
rem 1 min  0.226285  0.396679  0.311482
rem 3 min  0.186414  0.309328  0.247871
rem 5 min  0.165934  0.303076  0.234505

set PYTORCH_NVFUSER_DISABLE=1
set MULTIPROCESSING_START_METHOD=spawn
set OMP_NUM_THREADS=1

set PYTHON=D:\Applications\Anaconda\envs\LDWLip3\python.exe
if not exist "%PYTHON%" (
    echo LDWLip3 python not found: %PYTHON%
    pause
    exit /b 1
)

set ROOT_DIR=H:/Project/Lipreading/ldwLip_eaai
cd /d %ROOT_DIR%/code

set VAL_CMLR=code/labels/cmlr/cmlr_unseen_adapt_s2_1_test.csv
set CMLR_WEIGHTS=H:/Project/Lipreading/ldwLip_wjc/save/Baseline/train--Lipv1_cmlr_unseen/model_74.pth
if not exist "%CMLR_WEIGHTS%" (
    echo missing CMLR backbone: %CMLR_WEIGHTS%
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_VEcorr_cmlr.yaml" (
    echo missing VEcorr config
    pause
    exit /b 1
)

call :one lr010 0.010 || goto :fail
call :one lr015 0.015 || goto :fail
call :one lr020 0.020 || goto :fail

echo all 3 learning rates finished
pause
exit /b 0

:one
set TAG=%~1
set LR=%~2
for %%S in (s2 s6) do (
    for %%M in (1 3 5) do (
        echo ===== train %TAG% %%S %%M lr=%LR% epochs=7 =====
        "%PYTHON%" -u ldw_main.py vsr train ^
            data=cmlr_unseen.yaml ^
            model=PhD003/AdaptVSR3LoRA_VEcorr_cmlr.yaml ^
            root_dir=%ROOT_DIR% ^
            model_name=AdaptVSR3LoRA_VEcorr_%TAG%_%%S_%%M_cmlr ^
            train_data=code/labels/cmlr/cmlr_unseen_adapt_%%S_%%M_test.csv ^
            val_data=%VAL_CMLR% ^
            max_epochs=7 ^
            weights=%CMLR_WEIGHTS% ^
            warmup_epochs=0 ^
            lr=%LR% ^
            save_every_epoch=1 ^
            lora=True || exit /b 1
        for /L %%E in (0,1,6) do (
            echo ===== eval %TAG% %%S %%M model_%%E =====
            "%PYTHON%" -u ldw_main.py vsr eval ^
                data=cmlr_unseen.yaml ^
                model=PhD003/AdaptVSR3LoRA_VEcorr_cmlr.yaml ^
                root_dir=%ROOT_DIR% ^
                model_name=AdaptVSR3LoRA_VEcorr_%TAG%_%%S_%%M_ep%%E_cmlr ^
                test_data=code/labels/cmlr/cmlr_unseen_adapt_%%S_test.csv ^
                weights=%CMLR_WEIGHTS% ^
                lora_weights=output/train--AdaptVSR3LoRA_VEcorr_%TAG%_%%S_%%M_cmlr/model_%%E.pth ^
                lora=True || exit /b 1
        )
    )
)
exit /b 0

:fail
echo a VEcorr run failed
pause
exit /b 1
