#!/bin/bash

TARGET_DIR="./datasets"
mkdir -p "$TARGET_DIR"
BASE_URL="https://huggingface.co/datasets/latataro/sent-xrec-dataset/resolve/main"
FILES=("amazon_kindle_exps.pkl.gz")

for FILE in "${FILES[@]}"
do
    echo "Downloading $FILE..."
    curl -sL "${BASE_URL}/${FILE}" -o "${TARGET_DIR}/${FILE}"
done

echo "amazon_kindle file downloaded to $TARGET_DIR."