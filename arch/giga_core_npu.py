############################################
# GIGA Core NPU
# Author: Mohit Upadhyay
############################################

max_bw = 256

import numpy as np
from .giga_neuron_core import Giga_neuron_core
from .ps_router_giga import PS_router_giga
from .ws_router_giga import WS_router_giga
from .relu_unit import Relu_unit
from .training_systolic_array import Training_systolic_array
from .weight_update_engine import Weight_update_engine

# Nano core NPU design
class Giga_core_npu(Giga_neuron_core, PS_router_giga, WS_router_giga, Relu_unit, Training_systolic_array, Weight_update_engine):

	def __init__(self, input_size, output_size):
		# Microarchitecture declaration
		self.giga_neuron_core = Giga_neuron_core(input_size, output_size)

		self.ps_routers = [PS_router_giga() for i in range(max_bw)]
		self.ws_routers = [WS_router_giga() for i in range(max_bw)]
		self.relu_units = [Relu_unit() for i in range(max_bw)]

		self.training_systolic_array = Training_systolic_array()
		self.weight_update_engine = Weight_update_engine()

		# Control signals (specific to NPU) declaration
		self.inf_in_sel = 0
		self.tr_in_sel = 0
		self.sum_or_local = 0
		self.weight_update_en = 0
		self.mode = 0

	def load_weight_from_cpu(self, load_weight):

		self.giga_neuron_core.load_weight_from_cpu(load_weight)

	def forward_pass(self, ps_in_sel, ps_out_sel, ws_in_sel, ws_out_sel, bypass_en, sum_en, ps_en, inject_en, ws_bypass_en):
		
		# Control inputs
		self.mode = 0

		data_in = [0] * 256

		# Perform matrix multiplication
		self.giga_neuron_core.matrix_mul(mac_en=1, trans_en=0)

		# Routers to route PS and WS data
		self.ps_routers[0].route_in(ps_in_sel, bypass_en, sum_en)
		self.ps_routers[0].route_out(ps_out_sel, bypass_en, sum_en, ps_en)
		self.ws_routers[0].route_in(ws_in_sel, inject_en)
		self.ws_routers[0].route_out(ws_bypass_en, inject_en, ws_out_sel)
	
	def backward_pass(self, ps_in_sel, bypass_en, sum_en, ps_out_sel, ps_en, ws_in_sel, inject_en, ws_bypass_en, ws_out_sel):
		
		# Control inputs
		self.mode = 1

		data_in = [0] * 256

		# Perform matrix multiplication
		self.giga_neuron_core.matrix_mul(data_in, mac_en=0, trans_en=1)

		# Route the inference data into the 1D systolic array for weight grad calculation
		for row in range(max_bw):
			self.training_systolic_array.generate_weight_grad(data_in[row])

		# Routers to route PS and WS data
		self.ps_routers[0].route_in(ps_in_sel, bypass_en, sum_en)
		self.ps_routers[0].route_out(ps_out_sel, bypass_en, sum_en, ps_en)
		self.ws_routers[0].route_in(ws_in_sel, inject_en)
		self.ws_routers[0].route_out(ws_bypass_en, inject_en, ws_out_sel)

	def weight_update_operation(self, weight_update_en, learning_rate, weight_in):
		# Control Input
		self.weight_update_en = weight_update_en

		# Weight_update operation
		if (self.weight_update_en == 1):
			self.weight_update_engine.weight_update_operation(weight_in, learning_rate)

	def report_perf(self):

		cycles = self.giga_neuron_core.cycles + self.ps_routers[0].cycles + self.ws_routers[0].cycles
		load_cycles = self.giga_neuron_core.load_cycles

		return cycles, load_cycles