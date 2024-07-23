#!/bin/bash

VERSION="126.0.6478.182"
ARCH="linux64"

wget https://storage.googleapis.com/chrome-for-testing-public/$VERSION/$ARCH/chrome-headless-shell-linux64.zip -O chrome-headless-shell-$ARCH.zip 
wget https://storage.googleapis.com/chrome-for-testing-public/$VERSION/$ARCH/chromedriver-linux64.zip -O chromedriver-$ARCH.zip 
unzip chrome-headless-shell-linux64.zip -d ~/
unzip chromedriver-linux64.zip -d ~/
export PATH=~/chrome-headless-shell-linux64:~/chromedriver-linux64:$PATH