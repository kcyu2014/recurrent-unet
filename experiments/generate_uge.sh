#!/usr/bin/env bash

if [ ! -d "uge" ]; then
    mkdir ./uge
fi
#

if [ ! -z "$1" ];
then
    a=$1
else
    a='uge.sh'
fi

echo $a
cat $a | python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());" > ./uge/$a
chmod +x ./uge/$a
