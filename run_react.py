import argparse

from react_sim import react_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="../topologies/conv_nets/test.csv",
                        help="Path to the topology file"
                        )
    parser.add_argument('-num_giga_cores_x', metavar='GIGA core x', type=str,
                        default="4",
                        help="Num of GIGA cores x direction"
                        )
    parser.add_argument('-num_giga_cores_y', metavar='GIGA core y', type=str,
                        default="3",
                        help="Num of GIGA cores y direction"
                        )

    args = parser.parse_args()
    topofile = args.t
    num_giga_cores_x = args.num_giga_cores_x
    num_giga_cores_y = args.num_giga_cores_y

    s = react_sim()
    s.set_params(topofile=topofile, num_giga_cores_x=num_giga_cores_x, num_giga_cores_y=num_giga_cores_y)
    s.run()
    s.generate_output()