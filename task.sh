#!/bin/bash

# This script is used as a submitted task in Metacentrum

#PBS -N NN_training
#PBS -q gpu
#PBS -l scratch=300mb
#PBS -l walltime=2h
#PBS -l nodes=1:ppn=16:gpu=2
#PBS -l mem=4gb

## launch as: qsub task.sh

module add python-2.7.10-gcc
module add cuda-7.5
module add cudnn-5.0
module add caffe2016-gpu-cudnn

home="/storage/brno2/home/hacky"

trap 'clean_scratch' TERM EXIT
trap 'cp -r $SCRATCHDIR/log.out $SCRATCHDIR/*npz $SCRATCHDIR/trained $home && clean_scratch' TERM

#copy data
cp -r $home/arrays.py $home/boards2.npz $home/train2.py $home/trainQ.sh $home/queue $SCRATCHDIR
mkdir $SCRATCHDIR/trained

#launch task
cd $SCRATCHDIR

./trainQ.sh ./

#copy results to home
cp -r ./trained $home

#clean scratch
clean_scratch
