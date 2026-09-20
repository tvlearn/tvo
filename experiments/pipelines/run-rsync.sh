#!/bin/bash

source .config

rsync -avz --exclude experiments/out \
    $HOME/$projectpath \
    $username@$hostname:~/$projectpath
