@echo off
setlocal
rem Start this file by hand. It does not change ldwLip_wjc.
rem ldw_main.py prints straight to this window.
rem CVSR-Bili speakers b001 b017 b029 b032 b037 b040 b042 b052. Minutes 1, 3, 5.
rem Runs V-LoRA +Sem only.
rem Ours (w/o aug) is already stored under output\train--AdaptVSR3LoRA_NoAug_*_bili.
rem Those calls are commented below and are not run.
rem
rem 1. V-LoRA +Sem. Visual conv LoRA only, rank 4, alpha 8. No LoRA-MoE.
rem    Loss is id 0.15, corr 0.05, att 0.72, ctc 0.08.
rem    Identity is cross-entropy in the frozen speaker net, not the first two clips.
rem    lr 0.01, warmup 0, 7 epochs. Epochs are 0-based: model_0.pth through model_6.pth.
rem    In-loop validation uses b001 1-minute and is not the reported CER.
rem    Each saved adapter is evaluated on that speaker's full held-out list.
rem    Eval names end in _ep0 ... _ep6.
rem    Ignore result_all.txt global average cer.
rem    Read cer_total in output\test--<name>\log\result_on_device0.txt.
rem
rem 2. Ours (w/o aug). NoAug yaml. V-LoRA plus LoRAMoEPaper. No corpus mixup.
rem    Three experts, rank 4, alpha 8. Utterance router.
rem    Same loss weights, kept for all 6 epochs.
rem    Epochs 0-1 update expert 0 and lora_gate.
rem    Epochs 2-5 freeze expert 0 and update experts 1 and 2 together, plus lora_gate.
rem    Validation is that speaker's full held-out list.
rem    Read [Epoch 5] Global CER. The same list is evaluated again from model_5.pth.
rem    Again, read cer_total, not result_all.txt global average cer.

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

set JLW_WEIGHTS=H:/Project/Lipreading/ldwLip_wjc/save/Baseline/train--Lipv1_cvsrbili_unseen/model_74.pth
set SID_FILE=%ROOT_DIR%/save/sid/train--sid_meta/best.pth
set VAL_JLW=code/labels/cvsrbili/cvsrbili_unseen_adapt_b001_1_test.csv
if not exist "%JLW_WEIGHTS%" (
    echo missing cvsrbili backbone: %JLW_WEIGHTS%
    pause
    exit /b 1
)
if not exist "%SID_FILE%" (
    echo missing identity checkpoint: %SID_FILE%
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_VEid_cvsrbili.yaml" (
    echo missing VEid cvsrbili config
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_NoAug_cvsrbili.yaml" (
    echo missing NoAug cvsrbili config
    pause
    exit /b 1
)

set SID_WEIGHTS=%SID_FILE%
rem SID_LOSS=ce is the frozen speaker net. SID_LOSS=proto is the first-two-clip cosine.
set SID_LOSS=ce
rem For ce, SID_K_SHOT clips enroll each other speaker. SID_N_WAY counts the current clip too.
rem For proto, SID_K_SHOT is the number of leading adaptation clips in the prototype.
set SID_K_SHOT=2
set SID_N_WAY=8
set EAAI_AUX_DECAY=off
set EAAI_MOE_RANK_STEP=0
set EAAI_STAGE1_EPOCHS=0
set EAAI_EXPERT_EPOCHS=0

echo ===== V-LoRA +Sem identity-CE, CVSR-Bili, 7 epochs =====
for %%S in (b001 b017 b029 b032 b037 b040 b042 b052) do (
    for %%M in (1 3 5) do (
        call :sem %%S %%M || goto :fail
    )
)

rem Ours (w/o aug) already stored. Do not rerun.
rem set EAAI_EXPERT_EPOCHS=2
rem call :noaug b001 1
rem call :noaug b001 3
rem call :noaug b001 5
rem call :noaug b017 1
rem call :noaug b017 3
rem call :noaug b017 5
rem call :noaug b029 1
rem call :noaug b029 3
rem call :noaug b029 5
rem call :noaug b032 1
rem call :noaug b032 3
rem call :noaug b032 5
rem call :noaug b037 1
rem call :noaug b037 3
rem call :noaug b037 5
rem call :noaug b040 1
rem call :noaug b040 3
rem call :noaug b040 5
rem call :noaug b042 1
rem call :noaug b042 3
rem call :noaug b042 5
rem call :noaug b052 1
rem call :noaug b052 3
rem call :noaug b052 5

echo bili runs finished
if defined NO_PAUSE exit /b 0
pause
exit /b 0

:sem
set SPK=%~1
set MIN=%~2
set TRAINCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_%MIN%_test.csv
set TESTCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_test.csv
echo ===== train V-LoRA+Sem identity-CE bili %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=cvsrbili_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_VEid_cvsrbili.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_bili ^
    train_data=%TRAINCSV% ^
    val_data=%VAL_JLW% ^
    max_epochs=7 ^
    weights=%JLW_WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=1 ^
    lora=True
if errorlevel 1 exit /b 1
for /L %%E in (0,1,6) do (
    echo ===== eval V-LoRA+Sem identity-CE bili %SPK% %MIN%min model_%%E =====
    "%PYTHON%" -u ldw_main.py vsr eval ^
        data=cvsrbili_unseen.yaml ^
        model=PhD003/AdaptVSR3LoRA_VEid_cvsrbili.yaml ^
        root_dir=%ROOT_DIR% ^
        model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_ep%%E_bili ^
        test_data=%TESTCSV% ^
        weights=%JLW_WEIGHTS% ^
        lora_weights=output/train--AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_bili/model_%%E.pth ^
        lora=True
    if errorlevel 1 exit /b 1
)
exit /b 0

:noaug
set SPK=%~1
set MIN=%~2
set TRAINCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_%MIN%_test.csv
set TESTCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_test.csv
echo ===== train Ours w/o aug bili %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=cvsrbili_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_bili ^
    train_data=%TRAINCSV% ^
    val_data=%TESTCSV% ^
    max_epochs=6 ^
    weights=%JLW_WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=5 ^
    lora=True
if errorlevel 1 exit /b 1
echo ===== eval Ours w/o aug bili %SPK% %MIN%min model_5 =====
"%PYTHON%" -u ldw_main.py vsr eval ^
    data=cvsrbili_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_bili ^
    test_data=%TESTCSV% ^
    weights=%JLW_WEIGHTS% ^
    lora_weights=output/train--AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_bili/model_5.pth ^
    lora=True
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo stopped because a run failed
if defined NO_PAUSE exit /b 1
pause
exit /b 1
