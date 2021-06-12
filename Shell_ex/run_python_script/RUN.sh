#!/bin/bash

echo "Executing a bash statement"

set_env=$1
folder=$2

while read env; do echo $env; done < $folder/env_info.txt

$set_env/$env/bin/python $folder/test.py --env_path=$set_env --script_folder=$folder --input=$3 --target=$4 --env=$env
