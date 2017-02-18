#!/bin/bash

# This script trains all nets from  subdirectories
# from ./queue directory and stores results in
# subdirectories in ./trained directory

if [ -z $1 ]; then
	echo "Provide directory with nets and training script as argument!" && exit
fi

top_dir=$1

# omit / at the end
if [ ${1: -1} == "/" ]; then
	top_dir=${top_dir:0:-1}
fi

# check if directory with nets exists
if [ ! -d ${top_dir}/`ls ${top_dir} | grep "^queue$"` ]; then
    mkdir ${top_dir}/queue
    echo "Queue directory created."
    echo "Create directories each with net and solver you want to train."
    exit
fi

# create directory with results
mkdir -p ${top_dir}/trained

#iterate over all directories containing *.solverstate files (net and solver)
for net_dir in `ls ${top_dir}/queue`
do
	#get paths
	solver=`ls ${top_dir}/queue/${net_dir} | grep ".*solver.prototxt$"`

	net=`ls ${top_dir}/queue/${net_dir} | grep ".*val.prototxt$"`

	#copy *.solverstate files to location of training script
	cp ${top_dir}/queue/${net_dir}/${solver} ${top_dir}
	cp ${top_dir}/queue/${net_dir}/${net} ${top_dir}

    #switch solver mode to GPU (just in case)
    cat ${top_dir}/${solver} | sed 's/CPU/GPU/g' > ${top_dir}/${solver}

	#train net
	${top_dir}/train2.py -i=20000 -s=${top_dir}/${solver} -v

	if [ $? -ne "0" ]; then
		echo "Error occured during training!" && exit
	else
		#name of directory where to copy results
		res_dir=${top_dir}/trained/${net_dir}

        # create or clean directory containing results of computations
        if [ -d ${res_dir} ]; then
            rm -r ${res_dir}/*
        else
		    mkdir "${res_dir}"
        fi

		mv ${top_dir}/stats*.npz ${res_dir}
		mv ${top_dir}/*.solverstate ${res_dir}
		mv ${top_dir}/*.caffemodel ${res_dir}
		mv ${top_dir}/*.prototxt ${res_dir}

        # this directory exists on Metacentrum and
        # usually not on your PC
		if [ ! -d /storage ]; then
            #draw net
            ${top_dir}/draw_net.py ${top_dir}/queue/${net_dir}/net_train_val.prototxt net.svg

            mv ${top_dir}/probs*.svg ${res_dir}
            mv ${top_dir}/loss*.svg ${res_dir}
            mv ${top_dir}/net.svg ${res_dir}
        fi
	fi



done

