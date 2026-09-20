#!/bin/bash

# run the python script
T_START=$(date +'%F %T')
#echo "$T_START    Current git hash: $(git rev-parse HEAD)"
echo "$T_START    Job '$TASK_NAME' started"

LOCK_FILENAME=$OUT_PATH/.lock
if [ ! -f $LOCK_FILENAME ]; then
    mkdir -p $OUT_PATH
    echo "$(date +'%F %T')    Creating job lock file $LOCK_FILENAME"
    touch $LOCK_FILENAME

    DONE_FILENAME=$OUT_PATH/done.log
    FAILED_FILENAME=$OUT_PATH/failed.log

    if [ ! -f $DONE_FILENAME ]; then
        mkdir -p $OUT_PATH
        env > $OUT_PATH/.env

        source $TASK_SCRIPT_PATH
        exit_code=$?
        if [ $exit_code -eq 0 ]
        then
            FINISH_FILENAME=$DONE_FILENAME
        else
            FINISH_FILENAME=$FAILED_FILENAME
        fi
        T_END=$(date +'%F %T')
        echo "$T_END    Creating file $FINISH_FILENAME"
        touch $FINISH_FILENAME
        echo "$T_START    Start" >> $FINISH_FILENAME 
        echo "$T_END    End"     >> $FINISH_FILENAME 

    else
        echo "$(date +'%F %T')    Aborting, reason: found file $DONE_FILENAME"
    fi

    echo "$(date +'%F %T')    Job '$TASK_NAME' completed"
    echo "$(date +'%F %T')    Deleting job lock file"
    rm $LOCK_FILENAME
else
    echo "$(date +'%F %T')    Lock file found. Run 'make clean' if previous job crashed. Exiting"
fi

