############################################
# PS Router (Nano Core)
# Author: Mohit Upadhyay
############################################

filter_num = 64

from .nano_neuron_core import Nano_neuron_core

# Define the router behaviour
class PS_router(Nano_neuron_core):

	# Define the PS crossbar
	# Input 4 x 2 CB and then output crossbar 3 x 5
	def __init__(self):
		self.north_in = 0
		self.south_in = 0
		self.east_in = 0
		self.west_in = 0
		self.buffer_in = 0								# Additional input for the PS buffer

		self.bypass_out = 0    		 					# Bypass Path in the PS router
		self.local_ps = 0
		self.add_in_1 = 0			 					# Input for the adder (Local PS)
		self.add_in_2 = 0			 					# Input for the adder [OP2](from the 4 x 2 CB)
		self.add_out = 0       							# Output of the adder (from the 3 x 5 CB)
		self.buffer_out = [0] * 64						# Output to the PS buffer (64 x 256 for nano core)

		self.north_out = 0
		self.south_out = 0
		self.east_out = 0
		self.west_out = 0
		self.local_out = 0

		# Control Inputs
		self.in_sel = 0
		self.out_sel = 0
		self.bypass_en = 0
		self.sum_en = 0
		self.consec_add_en = 0
		self.ps_en = 0
		self.add_en = 0

		# Clock cycle consumption during ps router operation
		self.cycles = 0

	def reset_clock(self):
		self.cycles = 0

	def set_north_in(self, north_in_val):
		self.north_in = north_in_val

	def set_south_in(self, south_in_val):
		self.south_in = south_in_val

	def set_east_in(self, east_in_val):
		self.east_in = east_in_val

	def set_west_in(self, west_in_val):
		self.west_in = west_in_val

	# Routing the partial sum values into the router (set the operands for the addition operation)
	def route_in(self, in_sel, bypass_en, sum_en, consec_add_en, nano_neuron_core, index, filter_index):
		
		self.buffer_in = self.buffer_out[filter_index]

		self.local_ps = nano_neuron_core.data_out[index]

		# Control inputs
		self.in_sel = in_sel
		self.bypass_en = bypass_en
		self.sum_en = sum_en
		self.consec_add_en = consec_add_en

		if (self.consec_add_en == 1):
			self.add_in_1 = self.add_out

		else:
			self.add_in_1 = self.local_ps

		if (self.bypass_en == 1):
			if (self.in_sel == 0):
				self.bypass_out = self.north_in
			elif (self.in_sel == 1):
				self.bypass_out = self.south_in
			elif (self.in_sel == 2):
				self.bypass_out = self.east_in
			elif (self.in_sel == 3):
				self.bypass_out = self.west_in
			elif (self.in_sel == 4):
				self.bypass_out = self.buffer_in

		elif (self.sum_en == 1):
			if (self.in_sel == 0):
				self.add_in_2 = self.north_in
			elif (self.in_sel == 1):
				self.add_in_2 = self.south_in
			elif (self.in_sel == 2):
				self.add_in_2 = self.east_in
			elif (self.in_sel == 3):
				self.add_in_2 = self.west_in
			elif (self.in_sel == 4):
				self.add_in_2 = self.buffer_in

		# Cycles during routing in
		self.cycles = self.cycles + 1

	# Generate the full-weighted sum (to be routed to other cores/local output to Activation NoC)
	def gen_sum(self, add_en):

		# Control input
		self.add_en = add_en

		if (self.add_en == 1):
			self.add_out = self.add_in_1 + self.add_in_2

		# Cycles for ps addition
		self.cycles = self.cycles + 1

	def route_out(self, out_sel, bypass_en, sum_en, ps_en, filter_index):

		# Control inputs
		self.out_sel = out_sel
		self.bypass_en = bypass_en
		self.sum_en = sum_en
		self.ps_en = ps_en

		if (self.bypass_en == 1):
			if (self.out_sel == 0):
				self.north_out = self.bypass_out
			elif (self.out_sel == 1):
				self.south_out = self.bypass_out
			elif (self.out_sel == 2):
				self.east_out = self.bypass_out
			elif (self.out_sel == 3):
				self.west_out = self.bypass_out
			elif (self.out_sel == 4):
				self.local_out = 0
			elif (self.out_sel == 5):
				self.buffer_out[filter_index] = self.bypass_out

		elif (self.sum_en == 1):
			if (self.out_sel == 0):
				self.north_out = self.add_out
			elif (self.out_sel == 1):
				self.south_out = self.add_out
			elif (self.out_sel == 2):
				self.east_out = self.add_out
			elif (self.out_sel == 3):
				self.west_out = self.add_out
			elif (self.out_sel == 4):
				self.local_out = self.add_out
			elif (self.out_sel == 5):
				self.buffer_out[filter_index] = self.add_out

		elif (self.ps_en == 1):
			if (self.out_sel == 0):
				self.north_out = self.add_in_1
			elif (self.out_sel == 1):
				self.south_out = self.add_in_1
			elif (self.out_sel == 2):
				self.east_out = self.add_in_1
			elif (self.out_sel == 3):
				self.west_out = self.add_in_1
			elif (self.out_sel == 4):
				self.local_out = self.add_in_1
			elif (self.out_sel == 5):
				self.buffer_out[filter_index] = self.add_in_1

		# Cycles during routing out
		self.cycles = self.cycles + 1