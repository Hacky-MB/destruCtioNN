# destruCtioNN

This repository contains scripts which I use for training my neural networks for playing gomoku and other useful scripts/modules.

###File Description

arrays.py - module with operations with 2d over array

concat_stats.py	- script which concatenates more .npz files and recreates plots (useful when finetuning nets)

draw_net.py	- module for drawing nets (taken from [caffe python interface](http://caffe.berkeleyvision.org/tutorial/interfaces.html))

getBoards.py - script for parsing stored games and saving them in .npz format

play2.py - script for communication with game client and neural networks	

task.sh	- script for training using [Metacentrum](https://metavo.metacentrum.cz/en/)

train2.py - module for training neural network

trainQ.sh - script for training multiple networks one after another

###Dataset

Dataset was created by downloading AIs from [Gomocup website](http//:www.gomocup.org>) and letting them play using game client with each other. It consists from about 250 000 moves and can be downloaded [here](https://www.dropbox.com/s/atf2ts20nqeymno/boards2.npz?dl=0).

###Game client

The game client runs on windows and requires .exe file as AI. This file is located in "game client AI" folder and communicates with "play2.py" module using sockets. (IP address is hardcoded for now) I couldn't install caffe framework on windows so I run my game client on virtual machine and my python module on linux as a temporary solution.
