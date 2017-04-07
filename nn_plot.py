#!/usr/bin/env python

import numpy as np
import sys

class NNPlot:

	def __init__(self, show_plot, save_plot):
		self.iterations = np.array([])  # number of iterations passed
		self.test_guess = np.array([], dtype=float)  # correct guess (0-1) on test set
		self.test_guess_n = np.array([], dtype=float)  # guess in tom 5 most probable on test set
		self.train_guess = np.array([], dtype=float)  # correct guess (0-1) on train set
		self.train_guess_n = np.array([], dtype=float)  # guess in tom 5 most probable on train set
		self.loss = np.array([], dtype=float)  # output of loss function layer

		self.save_plot = save_plot
		self.show_plot = show_plot

		# number of points on plot
		self.points = 50

	def add_iter(self, it, test, test_n, train, train_n, curr_loss):
		self.iterations = np.append(self.iterations, it)
		self.test_guess = np.append(self.test_guess, test)
		self.test_guess_n = np.append(self.test_guess_n, test_n)
		self.train_guess = np.append(self.train_guess, train)
		self.train_guess_n = np.append(self.train_guess_n, train_n)
		self.loss = np.append(self.loss, curr_loss)

	def load_data(self, it, test, test_n, train, train_n, loss):
		self.iterations = it
		self.test_guess = test
		self.test_guess_n = test_n
		self.train_guess = train
		self.train_guess_n = train_n
		self.loss = loss

	def draw(self):
		import matplotlib.pyplot as plt
		import matplotlib.patches as mpatches
		length = len(self.iterations)

		# number of values averaged per one point in plot
		samples = length / self.points
		if not samples:
			samples = 1

		trash = length % samples

		# cut off values from the end so that length = points * samples
		if trash > 0:
			self.iterations = np.array(self.iterations[:-trash])
			self.test_guess = np.array(self.test_guess[:-trash])
			self.test_guess_n = np.array(self.test_guess_n[:-trash])
			self.train_guess = np.array(self.train_guess[:-trash])
			self.train_guess_n = np.array(self.train_guess_n[:-trash])
			self.loss = np.array(self.loss[:-trash])

		# samples * point
		length = len(self.iterations)

		# TODO: add train values
		(x, y1, y2) = (np.array(self.iterations[::length / self.points]),
						np.mean(self.train_guess.reshape(-1, samples), axis=1),
						np.mean(self.train_guess_n.reshape(-1, samples), axis=1)) \
			if length > self.points else (self.iterations, self.train_guess, self.train_guess_n)

		plt.title('Accuracy of test net')

		plt.plot(x, y1, linestyle='-', marker='o', color='red')
		plt.plot(x, y2, linestyle='-', marker='o', color='blue')
		red = mpatches.Patch(color='red', label='move guessed')
		blue = mpatches.Patch(color='blue', label='move in top 5')
		plt.legend(handles=[red, blue], loc=2, borderaxespad=0.)
		plt.ylim((0, 1))
		plt.xlim((0, x[-1]) if len(x) else (0, 1))

		# first save, then show (or save figure - fig = plt.gcf();fig.savefig('awd'))
		# new figure is created when show() is called
		if self.save_plot:
			plt.savefig('probs' + str(int(self.iterations[-1]) if length else "") + '.svg')

		if self.show_plot:
			plt.show()
		else:
			plt.clf()

		self.loss = np.mean(self.loss.reshape(-1, samples), axis=1) if length > self.points else self.loss

		plt.plot(x, self.loss, linestyle='-', marker='o', color='red')
		red = mpatches.Patch(color='red', label='loss')
		plt.legend(handles=[red], loc=2, borderaxespad=0.)
		plt.ylim((0, max(self.loss) + max(self.loss) * 0.1 if length else 1))
		plt.xlim((0, self.iterations[-1]) if length else (0, 1))

		if self.save_plot:
			plt.savefig('loss' + str(int(self.iterations[-1]) if length else "") + '.svg')

		if self.show_plot:
			plt.show()
		else:
			plt.clf()


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True)
	parser.add_argument('-p', '--show_plots', action='store_true')
	parser.add_argument('-r', '--save_plots', action='store_true')

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	return parser.parse_args()


def main():
	args = parse_args()

	f = np.load(args.input)
	plot = NNPlot(args.show_plots, args.save_plots)
	plot.load_data(f['iter'], [], [], f['train'], f['train_n'], f['loss'])
	plot.draw()

if __name__ == "__main__":
	main()
