#!/bin/bash

REPO_URL="https://github.com/nogi2k2/QML.git"
FOLDER_NAME="QML"

echo "Cloning Repository"
git clone "$REPO_URL"

cd "$FOLDER_NAME" 

echo "Pulling Git LFS files"
git lfs pull

echo "Clone Complete"
