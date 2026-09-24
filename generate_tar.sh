#!/bin/bash

ROOT="/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/data_cams"

cd "${ROOT}" || exit 1

for DIR in \
    deit_sat_tiny_patch16_224-sat-bcl-camelyon17_512_*_cams_train \
    resnet50-deepmil-bcl-camelyon17_512_*_cams_train \
    resnet50-pixelcam-bcl-camelyon17_512_*_cams_train
do

    # Vérifie que c'est bien un dossier
    if [ ! -d "${DIR}" ]; then
        continue
    fi

    OUTPUT="${DIR}.tar.gz"

    echo "============================================================"
    echo "Compressing:"
    echo "  ${DIR}"
    echo "-> ${OUTPUT}"
    echo "============================================================"

    # Skip si l'archive existe déjà
    if [ -f "${OUTPUT}" ]; then
        echo "[SKIP] ${OUTPUT} already exists."
        echo ""
        continue
    fi

    tar -czf "${OUTPUT}" "${DIR}"

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${OUTPUT}"
    else
        echo "[FAILED] ${DIR}"
    fi

    echo ""

done

echo "============================================================"
echo "ALL ARCHIVES COMPLETED"
echo "============================================================"