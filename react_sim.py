############################################
# Generate the different logical partitions for a single layer
# To generate the training and test data
# Author: Mohit Upadhyay
############################################

import numpy as np

from utils.topology_utils import topologies as topo
from mapper.single_layer_sim import single_layer_sim as layer_sim

# Core architectural features
max_dim = 256
max_img_dim = 16
max_output_channels = 64

class react_sim():

    def __init__(self):
        self.topo = topo()

        self.num_layers = 0
        self.single_layer_sim_object_list = []

        self.num_giga_cores_x = 0
        self.num_giga_cores_y = 0

        self.load_cycles = 0
        self.total_cycles = 0

    def set_params(self, topofile, num_giga_cores_x, num_giga_cores_y):

        self.topo.load_layer_params(topofile=topofile)

        # Calculate number of layers parameters here
        self.num_layers = self.topo.get_num_layers()

        self.num_giga_cores_x = int(num_giga_cores_x)
        self.num_giga_cores_y = int(num_giga_cores_y)

        print(self.num_giga_cores_x)
        print(self.num_giga_cores_y)
    
    def run(self):

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            this_layer_sim.set_params_single_layer(layer_id=i, entries=self.topo.topo_arrays)

            self.single_layer_sim_object_list.append(this_layer_sim)

        # 2. Run each layer simulation
        for single_layer_obj in self.single_layer_sim_object_list:

            layer_id = single_layer_obj.get_layer_id()
            layer_name = single_layer_obj.layer_name
            print('\n---------- Running Layer ' + str(layer_id) + ": " + str(layer_name) + " ----------\n")
            print("Layer_type, Ifmap h, ifmap w, filter h, filter w, num_ch, num_filt, stride (configuration values for each layer)")
            print(self.topo.topo_arrays[layer_id])

            single_layer_obj.run_single_layer(self.num_giga_cores_x, self.num_giga_cores_y)

            # Calculate the total number of cycles for the entire neural network
            self.load_cycles += single_layer_obj.load_cycles
            self.total_cycles += single_layer_obj.cycles

    def generate_output(self):

        print("\n************************************")
        print("Cycles to load weights: " + str(self.load_cycles))
        print("\nTotal number of cycles: " + str(self.total_cycles))

    def generate_graphs(self):

        return None