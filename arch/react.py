############################################
# REACT chip with 12 GIGA cores
# Author: Mohit Upadhyay
############################################

max_bw = 256
# num_nano_cores_x = 0
# num_nano_cores_y = 0
# num_giga_cores_x = 4
# num_giga_cores_y = 3

import numpy as np

from .giga_core_npu import Giga_core_npu

# Instantiation of a REACT architecture
class React(Giga_core_npu):

	def __init__(self, input_size, output_size, num_giga_cores_x, num_giga_cores_y):
		# Instantiate neuron core, ps routers, relu unit & activation routers
		self.num_giga_cores_x = num_giga_cores_x
		self.num_giga_cores_y = num_giga_cores_y

		self.giga_cores = [[Giga_core_npu(input_size, output_size) for j in range(self.num_giga_cores_y)] for i in range(self.num_giga_cores_x)]

	def load_weight_from_cpu(self, load_weight):

		for i in range(self.num_giga_cores_x):
			for j in range(self.num_giga_cores_y):

				self.giga_cores[i][j].load_weight_from_cpu(load_weight=load_weight)

	def forward_pass(self, ps_in_sel, ps_out_sel, ws_in_sel, ws_out_sel, bypass_en, sum_en, ps_en, inject_en, ws_bypass_en):

		for i in range(self.num_giga_cores_x):
			for j in range(self.num_giga_cores_y):

				self.giga_cores[i][j].forward_pass(ps_in_sel[i][j], ps_out_sel[i][j], ws_in_sel[i][j], ws_out_sel[i][j],
				       bypass_en[i][j], sum_en[i][j], ps_en[i][j], inject_en[i][j], ws_bypass_en[i][j])

	def backward_pass(self, ps_in_sel, bypass_en, sum_en, ps_out_sel, ps_en, ws_in_sel, inject_en, ws_bypass_en, axon_buffer_en, ws_out_sel):

		for i in range(self.num_giga_cores_x):
			for j in range(self.num_giga_cores_y):

				self.giga_cores[i][j].backward_pass(ps_in_sel[i][j], bypass_en[i][j], sum_en[i][j], ps_out_sel[i][j], ps_en[i][j], ws_in_sel[i][j], 
					inject_en[i][j], ws_bypass_en[i][j], axon_buffer_en[i][j], ws_out_sel[i][j])

	def report_perf(self):

		return self.giga_cores[0][0].report_perf()