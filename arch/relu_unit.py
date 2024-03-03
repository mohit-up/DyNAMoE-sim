############################################
# ReLU Unit
# Author: Mohit Upadhyay
############################################

# ReLU Unit implementation
class Relu_unit():

	#Define the relu unit
	def __init__(self):
		# Control input
		self.relu_en = 0

		# Data input
		self.input = 0
		self.relu_output = 0

	def relu_comp(self, relu_en, relu_input):
		# Control input
		self.relu_en = relu_en

		self.input = relu_input
		if (self.relu_en == 1):
			self.relu_output = self.input if (self.input > 0) else 0

		else:
			self.relu_output = self.input