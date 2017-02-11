#!/bin/bash

# This script trains all nets in ./queue directory 

if [ -z $1 ]; then
	echo "Provide directory with nets and training script as argument!" && exit
fi

top_dir=$1

# omit / at the end
if [ ${1: -1} == "/" ]; then
	top_dir=${top_dir:0:-1}
fi

#get name of directory of last trained net
res_dir_name=`ls "${top_dir}/trained" | tail -n1`

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

	#train net
	${top_dir}/train2.py -i=10000 -s=${top_dir}/queue/${net_dir}/${solver} -v

	exit

	if [ $? -ne "0" ]; then
		echo "Error occured while training!" #&& exit
	else
		#name of directory where to copy results
		res_dir_name=$((res_dir_name + 1))
		res_dir=${top_dir}/trained/${res_dir_name}

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

		#rm -r ${top_dir}/queue/${net_dir}
		rm ${top_dir}/net.svg
	fi

	rm ${top_dir}/probs*.svg
	rm ${top_dir}/loss*.svg
	rm ${top_dir}/stats*.npz
	rm ${top_dir}/*.solverstate
	rm ${top_dir}/*.caffemodel
	rm ${top_dir}/*.prototxt

done

