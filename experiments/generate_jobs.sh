#!/usr/bin/env bash

#if [ -d "multijobs" ]; then
#    rm -r ./multijobs
#fi

#mkdir ./multijobs
if [ ! -z "$1" ];
then
    a=$1
else
    a='multiple_jobs.sh'
fi

echo $a
cat $a | python3 -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());" > ./multijobs/$a
chmod +x ./multijobs/$a
