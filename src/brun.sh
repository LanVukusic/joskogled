#!/bin/bash

rm abc
echo start > abc
sbatch run.sh |
tail -f abc
