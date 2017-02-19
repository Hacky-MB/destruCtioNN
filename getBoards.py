#!/usr/bin/env python

import numpy as np
import sys
import os

"""
This script is used for parsing stored games and saving them in .npz format
"""

def parseArgs():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output', required=True, help='name of output file without ending')
	parser.add_argument('-i', '--input', required=True, help='directory containing .psq files')
	return parser.parse_args()


def main():
	args = parseArgs()

	files = [f for f in os.listdir(args.input) if  (len(f) > 4 and f[-3:] == "psq")]

	moves = []
	boards = []

	for file in files:
		with open(args.input+"/"+file.strip(), 'r') as f:
			lines = f.readlines()

			# parse only 19x19 games
			if "19x19" in lines[0]:
				board = np.zeros((19, 19))

				# skip first and last 3 lines
				for move in lines[1:-3]:
					move = move.split(',')
					if len(move) != 3:
						break

					# coordinates format in file: x,y,ms; x,y !@#$%^&*()_=+-_ <1,19>
					x, y = [int(x)-1 for x in move[:-1]]
					boards.append(np.copy(board).astype(np.uint8))
					moves.append(np.asarray((y,x)))
					board[y, x] = 1
					board[board > 0] = 1 - (board[board > 0] - 2)

	moves = np.stack(moves, axis=0)
	boards = np.stack(boards, axis=0)

	np.savez(args.output, boards=boards, moves=moves)


if __name__ == "__main__":
	main()
