@echo off
setlocal
rem Start this file by hand. It does not change ldwLip_wjc.
rem ldw_main.py prints straight to this window.
rem CMLR speakers s2 s6. Minutes 1, 3, 5.
rem Runs V-LoRA +Sem only.
rem Ours (w/o aug) is already stored under output\train--AdaptVSR3LoRA_NoAug_*_cmlr.
rem Those calls are commented below and are not run.
rem
rem 1. V-LoRA +Sem. Visual conv LoRA only, rank 4, alpha 8. No LoRA-MoE.
rem    Loss is id 0.15, corr 0.05, att 0.72, ctc 0.08.
rem    Identity is cross-entropy in the frozen speaker net, not the first two clips.
rem    lr 0.01, warmup 0, 7 epochs. Epochs are 0-based: model_0.pth through model_6.pth.
rem    In-loop validation uses s2 1-minute and is not the reported CER.
rem    Each saved adapter is evaluated on that speaker's full held-out list.
rem    Eval names end in _ep0 ... _ep6.
rem    For CMLR, result_on_device0.txt cer_total is the reported CER.
rem
rem 2. Ours (w/o aug). NoAug yaml. V-LoRA plus LoRAMoEPaper. No corpus mixup.
rem    Three experts, rank 4, alpha 8. Utterance router.
rem    Same loss weights, kept for all 6 epochs.
rem    Epochs 0-1 update expert 0 and lora_gate.
rem    Epochs 2-5 freeze expert 0 and update experts 1 and 2 together, plus lora_gate.
rem    Validation is that speaker's full held-out list.
rem    Read [Epoch 5] Global CER. The same list is evaluated again from model_5.pth.

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

set CMLR_WEIGHTS=H:/Project/Lipreading/ldwLip_wjc/save/Baseline/train--Lipv1_cmlr_unseen/model_74.pth
set SID_FILE=%ROOT_DIR%/save/sid/train--sid_meta/best.pth
set VAL_CMLR=code/labels/cmlr/cmlr_unseen_adapt_s2_1_test.csv
if not exist "%CMLR_WEIGHTS%" (
    echo missing CMLR backbone: %CMLR_WEIGHTS%
    pause
    exit /b 1
)
if not exist "%SID_FILE%" (
    echo missing identity checkpoint: %SID_FILE%
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_VEid_cmlr.yaml" (
    echo missing VEid cmlr config
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_NoAug_cmlr.yaml" (
    echo missing NoAug cmlr config
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

echo ===== V-LoRA +Sem identity-CE, CMLR, 7 epochs =====
for %%S in (s2 s6) do (
    for %%M in (1 3 5) do (
        call :sem %%S %%M || goto :fail
    )
)

rem Ours (w/o aug) already stored. Do not rerun.
rem set EAAI_EXPERT_EPOCHS=2
rem call :noaug s2 1
rem call :noaug s2 3
rem call :noaug s2 5
rem call :noaug s6 1
rem call :noaug s6 3
rem call :noaug s6 5

echo cmlr runs finished
if defined NO_PAUSE exit /b 0
pause
exit /b 0

:sem
set SPK=%~1
set MIN=%~2
set TRAINCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_%MIN%_test.csv
set TESTCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_test.csv
echo ===== train V-LoRA+Sem identity-CE cmlr %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=cmlr_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_VEid_cmlr.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_cmlr ^
    train_data=%TRAINCSV% ^
    val_data=%VAL_CMLR% ^
    max_epochs=7 ^
    weights=%CMLR_WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=1 ^
    lora=True
if errorlevel 1 exit /b 1
for /L %%E in (0,1,6) do (
    echo ===== eval V-LoRA+Sem identity-CE cmlr %SPK% %MIN%min model_%%E =====
    "%PYTHON%" -u ldw_main.py vsr eval ^
        data=cmlr_unseen.yaml ^
        model=PhD003/AdaptVSR3LoRA_VEid_cmlr.yaml ^
        root_dir=%ROOT_DIR% ^
        model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_ep%%E_cmlr ^
        test_data=%TESTCSV% ^
        weights=%CMLR_WEIGHTS% ^
        lora_weights=output/train--AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_cmlr/model_%%E.pth ^
        lora=True
    if errorlevel 1 exit /b 1
)
exit /b 0

:noaug
set SPK=%~1
set MIN=%~2
set TRAINCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_%MIN%_test.csv
set TESTCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_test.csv
echo ===== train Ours w/o aug cmlr %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=cmlr_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_NoAug_cmlr.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_cmlr ^
    train_data=%TRAINCSV% ^
    val_data=%TESTCSV% ^
    max_epochs=6 ^
    weights=%CMLR_WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=5 ^
    lora=True
if errorlevel 1 exit /b 1
echo ===== eval Ours w/o aug cmlr %SPK% %MIN%min model_5 =====
"%PYTHON%" -u ldw_main.py vsr eval ^
    data=cmlr_unseen.yaml ^
    model=PhD003/AdaptVSR3LoRA_NoAug_cmlr.yaml ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_cmlr ^
    test_data=%TESTCSV% ^
    weights=%CMLR_WEIGHTS% ^
    lora_weights=output/train--AdaptVSR3LoRA_NoAug_%SPK%_%MIN%_cmlr/model_5.pth ^
    lora=True
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo stopped because a run failed
if defined NO_PAUSE exit /b 1
pause
exit /b 1
