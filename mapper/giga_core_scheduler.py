############################################
# GIGA core scheduling
# Author: Mohit Upadhyay
############################################

max_dim = 256

# num_nano_cores_y = 0
# num_nano_cores_x = 0
# num_giga_cores_x = 4
# num_giga_cores_y = 3

import math
import sys

# Giga core scheduler
def schedule_giga(input_size, output_size, row_div, column_div, num, num_giga_cores_x, num_giga_cores_y):

    ps_in_sel = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    ps_out_sel = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    ws_in_sel = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    ws_out_sel = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]

    bypass_en = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    sum_en = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    ps_en = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    inject_en = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]
    ws_bypass_en = [[0 for j in range(num_giga_cores_y)] for i in range(num_giga_cores_x)]

    num_logical_cores = row_div * column_div
    num_physical_cores = num_giga_cores_x * num_giga_cores_y

    # Different mapping cases

    # Set the routing control signals
    if (num_logical_cores < num_physical_cores):

        if ((row_div == 1) and column_div <= num_giga_cores_y):

            for i in range(row_div):
                for j in range(column_div):

                    ps_in_sel[i][j] = 0
                    ps_out_sel[i][j] = 1
                    ws_in_sel[i][j] = 2
                    ws_out_sel[i][j] = 3

                    sum_en[i][j] = 1

        elif ((row_div <= num_giga_cores_x) and (column_div <= num_giga_cores_y)):
        
            for i in range(row_div):
                for j in range(column_div):

                    ps_in_sel[i][j] = 0
                    ps_out_sel[i][j] = 1
                    ws_in_sel[i][j] = 2
                    ws_out_sel[i][j] = 3

                    sum_en[i][j] = 1

        elif ((row_div > num_giga_cores_x) and (column_div <= num_giga_cores_y)):

            core_num_x = 0
            core_num_y = 0

            while ((core_num_x * num_giga_cores_y + core_num_y)  < num_logical_cores):

                ps_in_sel[core_num_x][core_num_y] = 2
                ps_out_sel[core_num_x][core_num_y] = 3
                ws_in_sel[core_num_x][core_num_y] = 0
                ws_out_sel[core_num_x][core_num_y] = 1

                sum_en[core_num_x][core_num_y] = 1

                core_num_y += 1
                if (core_num_y == num_giga_cores_y):
                    core_num_y = 0
                    core_num_x += 1

        elif ((row_div <= num_giga_cores_x) and (column_div > num_giga_cores_y)):

            core_num_x = 0
            core_num_y = 0

            while ((core_num_x * num_giga_cores_y + core_num_y)  < num_logical_cores):

                ps_in_sel[core_num_x][core_num_y] = 2
                ps_out_sel[core_num_x][core_num_y] = 3
                ws_in_sel[core_num_x][core_num_y] = 0
                ws_out_sel[core_num_x][core_num_y] = 1

                sum_en[core_num_x][core_num_y] = 1

                core_num_y += 1
                if (core_num_y == num_giga_cores_y):
                    core_num_y = 0
                    core_num_x += 1

    else:

        num_giga_partitions = row_div * column_div
        num_sch = math.ceil(num_giga_partitions/(num_giga_cores_x * num_giga_cores_y))

        logical_core_num = 0

        if (num < num_sch):

            for i in range(num_giga_cores_x):
                for j in range(num_giga_cores_y):

                    ps_in_sel[i][j] = 0
                    ps_out_sel[i][j] = 1
                    ws_in_sel[i][j] = 2
                    ws_out_sel[i][j] = 3

                    sum_en[i][j] = 1

                    logical_core_num = logical_core_num + 1

        else:

            # remaining_logical_core = num_giga_partitions - (num_sch - 1) * (num_giga_cores_x * num_giga_cores_y)

            for i in range(num_giga_cores_x):
                for j in range(num_giga_cores_y):

                    if (logical_core_num == num_giga_partitions):
                        break
                    else:
                        ps_in_sel[i][j] = 0
                        ps_out_sel[i][j] = 1
                        ws_in_sel[i][j] = 2
                        ws_out_sel[i][j] = 3

                        sum_en[i][j] = 1

                        logical_core_num = logical_core_num + 1

    return ps_in_sel, ps_out_sel, ws_in_sel, ws_out_sel, bypass_en, sum_en, ps_en, inject_en, ws_bypass_en