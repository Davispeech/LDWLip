@echo off
setlocal
rem Start this file by hand. It does not start itself and does not change ldwLip_wjc.
rem CMLR first, then CVSR-Bili. Each dataset runs V-LoRA +Sem, then Ours (w/o aug).
rem Identity loss is SID_LOSS=ce: the frozen speaker network.
rem V-LoRA +Sem writes output\train--AdaptVSR3LoRA_VEidCE_*.
rem Ours (w/o aug) writes output\train--AdaptVSR3LoRA_NoAugCE_*.
rem Existing AdaptVSR3LoRA_VEsem_* and AdaptVSR3LoRA_NoAug_* results stay in place.
rem
rem V-LoRA +Sem: rank 4, alpha 8, no LoRA-MoE. lr 0.01, warmup 0, 7 epochs.
rem Loss stays id 0.15, corr 0.05, att 0.72, ctc 0.08.
rem Epochs are 0-based: model_0.pth through model_6.pth. Each one is evaluated.
rem In-loop validation is s2 or b001 1-minute and is not the reported CER.
rem Held-out CER is cer_total in result_on_device0.txt.
rem For CVSR-Bili, ignore result_all.txt global average cer.
rem
rem Ours (w/o aug): NoAug yaml, LoRAMoEPaper, no corpus mixup.
rem EAAI_EXPERT_EPOCHS=2. Epochs 0-1 update expert 0 and lora_gate.
rem Epochs 2-5 freeze expert 0 and update experts 1 and 2 together.
rem Six epochs, lr 0.01. Validation is that speaker's full held-out list.
rem Read [Epoch 5] Global CER, then the eval of model_5.pth.

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
set JLW_WEIGHTS=H:/Project/Lipreading/ldwLip_wjc/save/Baseline/train--Lipv1_cvsrbili_unseen/model_74.pth
set SID_FILE=%ROOT_DIR%/save/sid/train--sid_meta/best.pth
set VAL_CMLR=code/labels/cmlr/cmlr_unseen_adapt_s2_1_test.csv
set VAL_JLW=code/labels/cvsrbili/cvsrbili_unseen_adapt_b001_1_test.csv
if not exist "%CMLR_WEIGHTS%" (
    echo missing CMLR backbone: %CMLR_WEIGHTS%
    pause
    exit /b 1
)
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
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_VEid_cmlr.yaml" (
    echo missing VEid cmlr config
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_VEid_cvsrbili.yaml" (
    echo missing VEid cvsrbili config
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_NoAug_cmlr.yaml" (
    echo missing NoAug cmlr config
    pause
    exit /b 1
)
if not exist "ldw_cfg\model\PhD003\AdaptVSR3LoRA_NoAug_cvsrbili.yaml" (
    echo missing NoAug cvsrbili config
    pause
    exit /b 1
)

set SID_WEIGHTS=%SID_FILE%
set SID_LOSS=ce
set SID_K_SHOT=2
set SID_N_WAY=8
set EAAI_AUX_DECAY=off
set EAAI_MOE_RANK_STEP=0
set EAAI_STAGE1_EPOCHS=0
set EAAI_EXPERT_EPOCHS=0

echo ===== CMLR V-LoRA +Sem identity-CE =====
for %%S in (s2 s6) do (
    for %%M in (1 3 5) do (
        call :sem cmlr %%S %%M || goto :fail
    )
)

echo ===== CMLR Ours w/o aug identity-CE =====
for %%S in (s2 s6) do (
    for %%M in (1 3 5) do (
        call :noaug cmlr %%S %%M || goto :fail
    )
)

echo ===== CVSR-Bili V-LoRA +Sem identity-CE =====
for %%S in (b001 b017 b029 b032 b037 b040 b042 b052) do (
    for %%M in (1 3 5) do (
        call :sem bili %%S %%M || goto :fail
    )
)

echo ===== CVSR-Bili Ours w/o aug identity-CE =====
for %%S in (b001 b017 b029 b032 b037 b040 b042 b052) do (
    for %%M in (1 3 5) do (
        call :noaug bili %%S %%M || goto :fail
    )
)

echo cmlr and bili identity-CE runs finished
pause
exit /b 0

:sem
set TAG=%~1
set SPK=%~2
set MIN=%~3
set EAAI_EXPERT_EPOCHS=0
if /I "%TAG%"=="cmlr" (
    set DATA=cmlr_unseen.yaml
    set MODEL=PhD003/AdaptVSR3LoRA_VEid_cmlr.yaml
    set WEIGHTS=%CMLR_WEIGHTS%
    set VALDATA=%VAL_CMLR%
    set TRAINCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_%MIN%_test.csv
    set TESTCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_test.csv
    set SUFFIX=cmlr
) else (
    set DATA=cvsrbili_unseen.yaml
    set MODEL=PhD003/AdaptVSR3LoRA_VEid_cvsrbili.yaml
    set WEIGHTS=%JLW_WEIGHTS%
    set VALDATA=%VAL_JLW%
    set TRAINCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_%MIN%_test.csv
    set TESTCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_test.csv
    set SUFFIX=bili
)
echo ===== train V-LoRA+Sem identity-CE %TAG% %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=%DATA% ^
    model=%MODEL% ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_%SUFFIX% ^
    train_data=%TRAINCSV% ^
    val_data=%VALDATA% ^
    max_epochs=7 ^
    weights=%WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=1 ^
    lora=True
if errorlevel 1 exit /b 1
for /L %%E in (0,1,6) do (
    echo ===== eval V-LoRA+Sem identity-CE %TAG% %SPK% %MIN%min model_%%E =====
    "%PYTHON%" -u ldw_main.py vsr eval ^
        data=%DATA% ^
        model=%MODEL% ^
        root_dir=%ROOT_DIR% ^
        model_name=AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_ep%%E_%SUFFIX% ^
        test_data=%TESTCSV% ^
        weights=%WEIGHTS% ^
        lora_weights=output/train--AdaptVSR3LoRA_VEidCE_%SPK%_%MIN%_%SUFFIX%/model_%%E.pth ^
        lora=True
    if errorlevel 1 exit /b 1
)
exit /b 0

:noaug
set TAG=%~1
set SPK=%~2
set MIN=%~3
set EAAI_EXPERT_EPOCHS=2
if /I "%TAG%"=="cmlr" (
    set DATA=cmlr_unseen.yaml
    set MODEL=PhD003/AdaptVSR3LoRA_NoAug_cmlr.yaml
    set WEIGHTS=%CMLR_WEIGHTS%
    set TRAINCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_%MIN%_test.csv
    set TESTCSV=code/labels/cmlr/cmlr_unseen_adapt_%SPK%_test.csv
    set SUFFIX=cmlr
) else (
    set DATA=cvsrbili_unseen.yaml
    set MODEL=PhD003/AdaptVSR3LoRA_NoAug_cvsrbili.yaml
    set WEIGHTS=%JLW_WEIGHTS%
    set TRAINCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_%MIN%_test.csv
    set TESTCSV=code/labels/cvsrbili/cvsrbili_unseen_adapt_%SPK%_test.csv
    set SUFFIX=bili
)
echo ===== train Ours w/o aug identity-CE %TAG% %SPK% %MIN%min =====
"%PYTHON%" -u ldw_main.py vsr train ^
    data=%DATA% ^
    model=%MODEL% ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAugCE_%SPK%_%MIN%_%SUFFIX% ^
    train_data=%TRAINCSV% ^
    val_data=%TESTCSV% ^
    max_epochs=6 ^
    weights=%WEIGHTS% ^
    warmup_epochs=0 ^
    lr=0.01 ^
    save_every_epoch=5 ^
    lora=True
if errorlevel 1 exit /b 1
echo ===== eval Ours w/o aug identity-CE %TAG% %SPK% %MIN%min model_5 =====
"%PYTHON%" -u ldw_main.py vsr eval ^
    data=%DATA% ^
    model=%MODEL% ^
    root_dir=%ROOT_DIR% ^
    model_name=AdaptVSR3LoRA_NoAugCE_%SPK%_%MIN%_%SUFFIX% ^
    test_data=%TESTCSV% ^
    weights=%WEIGHTS% ^
    lora_weights=output/train--AdaptVSR3LoRA_NoAugCE_%SPK%_%MIN%_%SUFFIX%/model_5.pth ^
    lora=True
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo stopped because a run failed
pause
exit /b 1
