#!/bin/bash

if [ -z "$1" ] 
then
    echo "$0: !!! Requies script as first argument !!!"
    exit 1
fi

if [ -z "$2" ] 
then
    echo "$0: !!! Requies make target as second argument !!!"
    exit 1
fi


echo "    Running all jobs..."
for file in $(find $1 -name \*.sh) ; do
    if [[ $(dirname $file) != "." ]]
    then
        echo "    Running make ${@:2} TASK_SCRIPT=$file"
        make "${@:2}" TASK_SCRIPT=$file 
    fi
done
