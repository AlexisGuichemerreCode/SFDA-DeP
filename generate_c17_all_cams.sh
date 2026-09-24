#!/bin/bash

# ============================================================
# Generate CAMs for all CAMELYON17 inter-center scenarios
#
# Methods:
#   - PixelCAM
#   - SAT
#   - DeepMIL
#
# Scenarios:
#   0 -> 1,2,3,4
#   1 -> 0,2,3,4
#   2 -> 0,1,3,4
#   3 -> 0,1,2,4
#   4 -> 0,1,2,3
# ============================================================


CUDA_ID=0
SPLIT="train"
CHECKPOINT_TYPE="best_classification"

DATASET="CAMELYON17_512"

SRC_MODEL_ROOT="/export/livia/home/vision/Aguichemerre/src_models_ICLR_2027"

CAM_ROOT="/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/data_cams"

GET_CAM_SCRIPT="get_cam.py"


# ============================================================
# PixelCAM source models
# ============================================================

declare -A PIXELCAM_MODELS

PIXELCAM_MODELS[0]="id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"

PIXELCAM_MODELS[1]="id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"

PIXELCAM_MODELS[2]="id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"

PIXELCAM_MODELS[3]="id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"

PIXELCAM_MODELS[4]="id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"


# ============================================================
# SAT source models
# ============================================================

declare -A SAT_MODELS

SAT_MODELS[0]="id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224"

SAT_MODELS[1]="id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224"

SAT_MODELS[2]="id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224"

SAT_MODELS[3]="id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224"

SAT_MODELS[4]="id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224"


# ============================================================
# DeepMIL source models
# ============================================================

declare -A DEEPMIL_MODELS

DEEPMIL_MODELS[0]="id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50"

DEEPMIL_MODELS[1]="id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50"

DEEPMIL_MODELS[2]="id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50"

DEEPMIL_MODELS[3]="id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50"

DEEPMIL_MODELS[4]="id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50"


# ============================================================
# Create output directory
# ============================================================

mkdir -p "${CAM_ROOT}"


# ============================================================
# Check source models
# ============================================================

echo "============================================================"
echo "CHECKING SOURCE MODELS"
echo "============================================================"

for SRC in 0 1 2 3 4
do

    for METHOD in PIXELCAM SAT DEEPMIL
    do

        case ${METHOD} in

            PIXELCAM)
                MODEL="${PIXELCAM_MODELS[$SRC]}"
                ;;

            SAT)
                MODEL="${SAT_MODELS[$SRC]}"
                ;;

            DEEPMIL)
                MODEL="${DEEPMIL_MODELS[$SRC]}"
                ;;

        esac

        MODEL_PATH="${SRC_MODEL_ROOT}/${MODEL}"

        if [ ! -d "${MODEL_PATH}" ]; then

            echo "WARNING: model not found:"
            echo "${MODEL_PATH}"

        else

            echo "[OK] ${METHOD} source=${SRC}"

        fi

    done

done


echo ""
echo "============================================================"
echo "START CAM GENERATION"
echo "============================================================"


# ============================================================
# Main loops
# ============================================================

for METHOD in PIXELCAM SAT DEEPMIL
do

    # --------------------------------------------------------
    # Configure architecture depending on WSOL method
    # --------------------------------------------------------

    case ${METHOD} in

        PIXELCAM)

            ENCODER="resnet50"
            PIXEL_WISE="True"
            METHOD_NAME="pixelcam"

            ;;

        SAT)

            ENCODER="deit_sat_tiny_patch16_224"
            PIXEL_WISE="False"
            METHOD_NAME="sat"

            ;;

        DEEPMIL)

            ENCODER="resnet50"
            PIXEL_WISE="False"
            METHOD_NAME="deepmil"

            ;;

    esac


    echo ""
    echo "############################################################"
    echo "METHOD: ${METHOD}"
    echo "############################################################"


    for SOURCE_FOLD in 0 1 2 3 4
    do

        # ----------------------------------------------------
        # Get appropriate source model
        # ----------------------------------------------------

        case ${METHOD} in

            PIXELCAM)
                SOURCE_MODEL="${PIXELCAM_MODELS[$SOURCE_FOLD]}"
                ;;

            SAT)
                SOURCE_MODEL="${SAT_MODELS[$SOURCE_FOLD]}"
                ;;

            DEEPMIL)
                SOURCE_MODEL="${DEEPMIL_MODELS[$SOURCE_FOLD]}"
                ;;

        esac


        SOURCE_MODEL_PATH="${SRC_MODEL_ROOT}/${SOURCE_MODEL}"


        # ----------------------------------------------------
        # Loop over all target centers
        # ----------------------------------------------------

        for TARGET_FOLD in 0 1 2 3 4
        do

            # Do not generate source -> same source
            if [ "${SOURCE_FOLD}" -eq "${TARGET_FOLD}" ]; then
                continue
            fi


            # ------------------------------------------------
            # CAM output path
            # ------------------------------------------------

            PATH_CAM="${CAM_ROOT}/${ENCODER}-${METHOD_NAME}-bcl-camelyon17_512_${SOURCE_FOLD}-camelyon17_512_${TARGET_FOLD}_cams_${SPLIT}"


            echo ""
            echo "============================================================"
            echo "${METHOD}: C17-${SOURCE_FOLD} -> C17-${TARGET_FOLD}"
            echo "============================================================"
            echo "Encoder:      ${ENCODER}"
            echo "Pixel wise:   ${PIXEL_WISE}"
            echo "Source model: ${SOURCE_MODEL_PATH}"
            echo "CAM path:     ${PATH_CAM}"
            echo "============================================================"


            # ------------------------------------------------
            # Generate CAM
            # ------------------------------------------------

            python "${GET_CAM_SCRIPT}" \
                --cudaid "${CUDA_ID}" \
                --split "${SPLIT}" \
                --checkpoint_type "${CHECKPOINT_TYPE}" \
                --encoder_name "${ENCODER}" \
                --pixel_wise_classification "${PIXEL_WISE}" \
                --source_dataset "${DATASET}" \
                --fold_compute_cam "${TARGET_FOLD}" \
                --path_cam "${PATH_CAM}" \
                --path_pre_trained_source "${SOURCE_MODEL_PATH}"


            # ------------------------------------------------
            # Check command result
            # ------------------------------------------------

            EXIT_CODE=$?

            if [ ${EXIT_CODE} -eq 0 ]; then

                echo ""
                echo "[SUCCESS] ${METHOD}: C17-${SOURCE_FOLD} -> C17-${TARGET_FOLD}"

            else

                echo ""
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                echo "[FAILED] ${METHOD}: C17-${SOURCE_FOLD} -> C17-${TARGET_FOLD}"
                echo "Exit code: ${EXIT_CODE}"
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

            fi

        done

    done

done


echo ""
echo "============================================================"
echo "ALL CAM GENERATION COMMANDS COMPLETED"
echo "============================================================"