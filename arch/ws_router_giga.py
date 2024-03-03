############################################
# Activation Router
# Author: Mohit Upadhyay
############################################

# Define the router behaviour
class WS_router_giga():

	# Define the Activation crossbar
	# 6 x 6 Crossbar
	def __init__(self):
		self.north_in = 0
		self.south_in = 0
		self.east_in = 0
		self.west_in = 0
		self.local_in = 0
		self.buffer_in = 0

		self.cb_input = 0

		self.north_out = 0
		self.south_out = 0
		self.east_out = 0
		self.west_out = 0
		self.local_out = 0
		self.buffer_out = 0

		# Control Inputs
		self.in_sel = 0
		self.out_sel = 0
		self.inject_en = 0
		self.ws_bypass_en = 0

		# Clock cycle consumption during activation router operation
		self.cycles = 0

	def reset_clock(self):
		self.cycles = 0

	# def set_north_in(self, north_in_val):
	# 	self.north_in = north_in_val

	# def set_south_in(self, south_in_val):
	# 	self.south_in = south_in_val

	# def set_east_in(self, east_in_val):
	# 	self.east_in = east_in_val

	# def set_west_in(self, west_in_val):
	# 	self.west_in = west_in_val
		
	def route_in(self, in_sel, inject_en):

		# Control inputs
		self.in_sel = in_sel
		self.inject_en = inject_en

		if (self.inject_en == 0):
			if (self.in_sel == 0):
				self.cb_input = self.north_in
			elif (self.in_sel == 1):
				self.cb_input = self.south_in
			elif (self.in_sel == 2):
				self.cb_input = self.east_in
			elif (self.in_sel == 3):
				self.cb_input = self.west_in
			elif (self.in_sel == 4):
				self.cb_input = self.buffer_in

		else:
			self.cb_input = self.local_in

		# Cycles during routing in
		self.cycles = self.cycles + 1

	def route_out(self, ws_bypass_en, inject_en, out_sel):

		# Control inputs
		self.inject_en = inject_en
		self.out_sel = out_sel
		self.ws_bypass_en = ws_bypass_en

		if (self.ws_bypass_en == 1 or self.inject_en == 1):
			if (self.out_sel == 0):
				self.north_out = self.cb_input
			elif (self.out_sel == 1):
				self.south_out = self.cb_input
			elif (self.out_sel == 2):
				self.east_out = self.cb_input
			elif (self.out_sel == 3):
				self.west_out = self.cb_input
			elif (self.out_sel == 4):
				self.buffer_out = self.cb_input

		else:
			self.local_out = self.cb_input

		# Cycles during routing out
		self.cycles = self.cycles + 1