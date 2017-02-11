#import caffe
import time
import numpy as np
import sys
import os
import random
import math
import SocketServer

mine = 0
enemy = 1

def my_turn(board,x,y):
	exp = True
	while exp:
		x,y = random.randint(0,18),random.randint(0,18)

		print("generujem " + str(x) + " " + str(y) + " " + str(board[mine][x][y]))


		if board[mine][x][y] == False and board[enemy][x][y] == False:
			board[mine][x][y] = True
			print("nastavujem ["+str(x)+"]["+str(y)+"]")
			exp = False
	return board,x,y

class MyTCPHandler(SocketServer.BaseRequestHandler):
	"""
	The request handler class for our server.

	It is instantiated once per connection to the server, and must
	override the handle() method to implement communication to the
	client.
	"""

	def handle(self):
		board = [[[False for x in range(19)] for y in range(19)] for z in range(2)]
		while True:
			'''data_recieved = False
			first = True
			while not data_recieved:
				self.data = self.request.recv(20).strip()
				print("DATA: " + str(len(self.data)))
				if (len(self.data) != 2):
					if first:
						first = False
					self.request.sendall(bytearray("ko"))
				else:
					data_recieved = True'''

			exp = True
			while exp:
				self.data = self.request.recv(20).strip()
				if len(self.data):
					exp = False
					
			print("DATA: " + str(len(self.data)))
			for i in range(len(self.data)):
				print str(ord(self.data[i])),
			print("")
			x,y = ord(self.data[-2]), ord(self.data[-1])
			sys.stdout.flush()

			#initialize board
			if x == ord('i') and y == ord('n'):
				board = [[[False for a1 in range(19)] for a2 in range(19)] for a3 in range(2)]

			else:
				#start first
				if x == ord('s') and y == ord('t'):
					board,x,y = my_turn(board,x,y)

				#make enemy move and start
				else:
					board[enemy][x][y] = True
					board,x,y = my_turn(board,x,y)

				output = bytearray([x,y])
				self.request.send(output)
				
				print( "posielam " + str(x) + " " + str(y))
				print("")

if __name__ == "__main__":
	random.seed(time.clock())

	HOST, PORT = "192.168.56.1", 27015

	# Create the server, binding to localhost on port 
	server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

	# Activate the server; this will keep running until you
	# interrupt the program with Ctrl-C
	server.serve_forever()
