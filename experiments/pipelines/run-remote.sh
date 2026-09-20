#!/bin/bash

if [ -z "$1" ] 
then
    echo "$0: !!! Requires command as first argument !!!"
    exit 1
fi

source .config


rsync -avz --exclude experiments/out \
    $HOME/$projectpath \
    $username@$hostname:~/$projectpath


ssh $username@$hostname <<-ENDSSH
    cd ${PWD/#$HOME/\/user\/$username}
    echo "    Current path: \$PWD"
    echo "    Will run ${@:1}"
    ${@:1}
ENDSSH


