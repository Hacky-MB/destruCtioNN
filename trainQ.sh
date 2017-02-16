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
for net_dir in `ls ${top_dir}/queue` # 
do
	#get paths
	solver=`ls ${top_dir}/queue/${net_dir} | grep ".*solver.prototxt$"`
	#solver=`realpath ${top_dir}/queue/${net_dir}/$solver`

	net=`ls ${top_dir}/queue/${net_dir} | grep ".*val.prototxt$"`
	#net=`realpath ${top_dir}/queue/${net_dir}/$net`

	#copy *.solverstate files to location of training script
	cp ${top_dir}/queue/${net_dir}/${solver} ${top_dir}
	cp ${top_dir}/queue/${net_dir}/${net} ${top_dir}

    #switch solver mode to GPU (just in case)
    cat net_solver.prototxt | sed 's/CPU/GPU/g' > net_solver.prototxt

	#train net
	${top_dir}/train2.py -i=10000 -s=${top_dir}/queue/${net_dir}/${solver} -v

	if [ $? -ne "0" ]; then
		echo "Error occured while training!" && exit
	else
		#name of directory where to copy results
		res_dir=${top_dir}/trained/${net_dir}

		#draw net
		${top_dir}/draw_net.py ${top_dir}/queue/${net_dir}/net_train_val.prototxt net.svg

		mkdir "${res_dir}"
		cp ${top_dir}/probs*.svg ${res_dir}
		cp ${top_dir}/loss*.svg ${res_dir}
		cp ${top_dir}/stats*.npz ${res_dir}
		cp ${top_dir}/*.solverstate ${res_dir}
		cp ${top_dir}/*.caffemodel ${res_dir}
		cp ${top_dir}/*.prototxt ${res_dir}
		cp ${top_dir}/net.svg ${res_dir}

        rm ${top_dir}/probs*.svg
        rm ${top_dir}/loss*.svg
        rm ${top_dir}/stats*.npz
        rm ${top_dir}/*.solverstate
        rm ${top_dir}/*.caffemodel
        rm ${top_dir}/*.prototxt
		rm ${top_dir}/net.svg
	fi



done

