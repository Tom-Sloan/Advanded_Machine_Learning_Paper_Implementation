#!/bin/bash
FILE=$1

# Define valid dataset options
if [[ $FILE != "cityscapes" && $FILE != "night2day" && $FILE != "edges2handbags" && \
      $FILE != "edges2shoes" && $FILE != "facades" && $FILE != "maps" && \
      $FILE != "openwebtext" && $FILE != "shakespeare" ]]; then
    echo "Available datasets are: cityscapes, night2day, edges2handbags, edges2shoes, facades, maps, openwebtext, shakespeare"
    exit 1
fi

echo "Specified [$FILE]"

# Create datasets directory if it doesn't exist
DATASET_DIR="./datasets"
mkdir -p $DATASET_DIR

if [[ $FILE == "openwebtext" ]]; then
    echo "Preparing OpenWebText dataset..."
    python data/openwebtext/prepare.py
    
elif [[ $FILE == "shakespeare" ]]; then
    echo "Preparing Shakespeare datasets..."
    
    # Create directory structure
    mkdir -p $DATASET_DIR/shakespeare/char
    mkdir -p $DATASET_DIR/shakespeare/word
    
    # Run character-level dataset preparation
    echo "Creating character-level dataset..."
    python -m src.data_management.dataset ShakespeareCharDataset \
        --data_dir $DATASET_DIR/shakespeare/char
    
    # Run word-level dataset preparation
    echo "Creating word-level dataset..."
    python -m src.data_management.dataset ShakespeareWordDataset \
        --data_dir $DATASET_DIR/shakespeare/word
    
    echo "Shakespeare datasets prepared successfully"
    
else
    # Original pix2pix dataset download logic
    URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
    TAR_FILE=$DATASET_DIR/$FILE.tar.gz
    TARGET_DIR=$DATASET_DIR/$FILE/

    echo "Downloading dataset to $TAR_FILE..."
    wget -N $URL -O $TAR_FILE

    if [ -f "$TAR_FILE" ]; then
        echo "Extracting dataset..."
        mkdir -p $TARGET_DIR
        tar -zxvf $TAR_FILE -C $DATASET_DIR
        rm $TAR_FILE
        echo "Dataset extracted successfully to $TARGET_DIR"
    else
        echo "Failed to download dataset"
        exit 1
    fi
fi