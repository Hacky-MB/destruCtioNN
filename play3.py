#!/usr/bin/env python

import caffe
import time
import numpy as np
import socket
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import layer
import arrays
import sys
from neural_network import NeuralNetwork
from ai import AI

"""
This script is used for communicating with game client using sockets
It recieved coordinates, updates board state and sends next move to
AI run by game client.
"""


class NeuralNetworkArgs:
	def __init__(self, solver, snapshot, net_type):
		self.solver = solver
		self.load_snapshot = snapshot
		self.batch_size = 1
		self.net = net_type
		self.mode = NeuralNetwork.ComputingMode['CPU']


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--ip_addr', default="192.168.56.1")
	parser.add_argument('-l', '--snapshot', required=True)
	parser.add_argument('-p', '--port', default=27015, type=int)
	parser.add_argument('-s', '--solver', required=True)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	return parser.parse_args()


def disconnect(sock, connection):
	sock.shutdown(socket.SHUT_RDWR)
	sock.close()


def get_response(data):
	print type(data[0])
	for bit in data:
		print "\""+str(bit)+"\"",
	print
	for bit in data:
		print "\""+str(ord(bit))+"\"",
	print
	sys.stdout.flush()
	index = 0
	ret = []

	while index < len(data):
		#
		if ord(data[index]) == 255:
			index += 1
		else:
			ret.append(bytearray([]))

			while index < len(data):
				print ord(data[index]), index
				sys.stdout.flush()
				ret[-1].append(data[index])
				index += 1

				if index == len(data) or ord(data[index]) == 255:
					break
	return ret


def main():
	args = parse_args()
	host, port = args.ip_addr, args.port

	nn_args = NeuralNetworkArgs(args.solver, args.snapshot, NeuralNetwork.TrainedNet['Policy'])
	brain = AI(nn_args)

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind((host, port))
	sock.listen(2)
	connection = None

	loading_board = False

	try:
		while True:
			connection, client_address = sock.accept()
			while True:
				data = connection.recv(50).strip()

				# if connection was closed
				if not len(data):
					break

				print "-------------------------------------"
				data = get_response(data)
				#print "len - "+str(len(data))

				for command in data:
					# print "|"+command+"|"
					# for c in command:
					# 	print "\""+str(c)+"\"",
					# print
					x, y = command[0], command[1]

					# initialize board
					if x == ord('i') and y == ord('n'):
						brain.reset_board()

					# board (load preloaded board state)
					elif x == ord('b') and y == ord('o'):
						print "start board transfer"
						loading_board = True
						board = np.zeros((2, 19, 19))

					# done (loading board state)
					elif x == ord('d') and y == ord('o'):
						print "end board transfer"
						brain.load_board(board)
						loading_board = False

					# move
					elif x == ord('m') and y == ord('o'):
							print "making move"
							x, y = brain.make_move()
							output = bytearray([x, y])
							print "sending " + str(y) + " " + str(x)
							connection.sendall(output)
							print "move sent"

					# make enemy move and my move
					else:
						if loading_board:
							player = command[2]
							board[player, y, x] = 1
						else:
							brain.enemy_move(x, y)

			#disconnect(sock, connection)

	except KeyboardInterrupt:
		disconnect(sock, connection)

	except:
		print "Unexpected error:", sys.exc_info()[0]
		disconnect(sock, connection)
		raise


if __name__ == "__main__":
	main()
