############################################
# GIGA Neuron Core
# Author: Mohit Upadhyay
############################################

import numpy as np

max_bw = 256

# Defining the GIGA neuron core functionality
class Giga_neuron_core():
    
	def __init__(self, input_size, output_size):
		# Data inputs
		self.data_in = np.zeros((1, input_size))
		self.weight = np.zeros((input_size, output_size))            # Weight sent from the CPU/IO to the SRAM
		self.data_out = np.zeros((1, output_size))

		# Control Inputs
		self.mac_en = 0
		self.trans_en = 0
		self.load_weight = 0

		# Cycles required for neuron core operation
		self.load_cycles = 0
		self.cycles = 0

	def reset_clock(self):
		self.load_cycles = 0
		self.cycles = 0

	def load_weight_from_cpu(self, load_weight):
		self.load_weight = load_weight

		if (self.load_weight == 1):
			self.load_weight = 0

		# No of cycles for loading weight
		self.load_cycles = self.load_cycles + len(self.weight)
		self.cycles = self.cycles + len(self.weight)

	def matrix_mul(self, mac_en, trans_en):
		# Control Inputs
		self.mac_en = mac_en
		self.trans_en = trans_en

		# No of cycles for matrix mul operation (Forward/Backward Propagation)
		self.cycles = self.cycles + len(self.weight)