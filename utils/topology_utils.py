import math

class topologies(object):

    def __init__(self):
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_arrays = []
        self.num_layers = 0

    # reset topology parameters
    def reset(self):
        print("All data reset")
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_arrays = []
        self.num_layers = 0

    # Load topology param from topology file
    def load_layer_params(self, topofile=''):

        self.topo_file_name = topofile.split('/')[-1]
        name_arr = self.topo_file_name.split('.')
        if len(name_arr) > 1:
            self.current_topo_name = self.topo_file_name.split('.')[-2]
        else:
            self.current_topo_name = self.topo_file_name

        f = open(topofile, 'r')
        first = True

        # Get the parameters for different layers
        for row in f:
            row = row.strip()
            if first:
                first = False
                continue
            elif row == '':
                continue
            else:
                elems = row.split(',')[:-1]
                assert len(elems) > 8, 'There should be at least 9 entries per row'
                layer_name = elems[0].strip()
                layer_type = elems[1].strip()
                ifmap_height = int(elems[2].strip())
                ifmap_width = int(elems[3].strip())
                kernel_height = int(elems[4].strip())
                kernel_width = int(elems[5].strip())
                input_ch = int(elems[6].strip())
                output_ch = int(elems[7].strip())
                stride = int(elems[8].strip())

                # Entries: Layer_type, Ifmap h, ifmap w, filter h, filter w, num_ch, num_filt, stride (configuration values for each layer)
                entries = [layer_name, layer_type, ifmap_height, ifmap_width, kernel_height, kernel_width, input_ch, output_ch, stride]

                # Create a list of parameters for different layers
                self.topo_arrays.append(entries)
                # self.append_topo_arrays(layer_type=layer_type, elems=entries)

        self.num_layers = len(self.topo_arrays)

    def get_num_layers(self):
        return self.num_layers