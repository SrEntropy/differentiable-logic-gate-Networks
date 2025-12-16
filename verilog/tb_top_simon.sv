`timescale 1ns / 1ps

module tb_top_simon;

    parameter MAX_X_COORD = 240;
    parameter MAX_Y_COORD = 180;
    parameter INPUT_PATH = "C:/projects/npc25-difflogic-4-control/verilog/events.txt";
    parameter NS_PER_CLK = 5; // 200MHz is clk every 5ns
    parameter TIME_WINDOW = 100000; // We test only single time window
    parameter ADDR_OUT = $clog2(4*4*4);

    logic clk;
    logic rst;
    logic [(34*34*2)-1 : 0] out_data;
    logic [ADDR_OUT-1 : 0]  out_addr;
    logic out_valid;

    // Queues with values from file
	logic [33: 0]  timestamps [$];
	logic [13 : 0]  x_coords [$];
	logic [13 : 0]  y_coords [$];
	logic	       polarities [$];

    // Values read from queue
	logic [33: 0]  timestamp;
	logic [13 : 0]  x_coord;
	logic [13 : 0]  y_coord;
	logic	       polarity;
	logic [33: 0]  timestamp_reg;
	logic [13 : 0]  x_coord_reg;
	logic [13 : 0]  y_coord_reg;
	logic	       polarity_reg;
	logic          is_valid;

    // Input and output files handler, scheduler.
    int            cnt = 0;
    int            file;
    int            file_out;
    string         line;
    int            current_time_ns;
    string         x_string;
    string         y_string;
    string         t_string;
    string         p_string;
    
    int debug_file;
    initial begin
        debug_file = $fopen("C:/projects/npc25-difflogic-4-control/verilog/debug.txt", "w");
    end
    always @(posedge i_dut.i_output_accumulator_classifier.inp_valid_i) begin
        if (debug_file)
            $fwrite(debug_file, "%b\n", i_dut.i_diff_logic_network.out);
    end

    initial begin
        force i_dut.i_output_accumulator_classifier.class_moving_average_q[0] = 0;
    end

    initial begin
        file = $fopen(INPUT_PATH, "r");

        while(!$feof(file)) begin
            $fgets(line, file);
            $sscanf (line, "%s %s %s %s", x_string, y_string, t_string, p_string);
            
            // Save polarity as 1 or 0
            if (p_string == "1") polarities.push_back(1'b1);
            else polarities.push_back(1'b0);
            
            // Save timestamps, change unit to us
            timestamps.push_back(t_string.atoi());
            
            // Save coordinates
            x_coords.push_back(x_string.atoi());
            y_coords.push_back(y_string.atoi());

        end
        $fclose(file);
        //$display(timestamps.size());

        // Get first values from queue
        timestamp_reg = timestamps.pop_front();
        x_coord_reg   = x_coords.pop_front();
        y_coord_reg   = y_coords.pop_front();
        polarity_reg  = polarities.pop_front();

        while(1) begin
            if (cnt<10) begin
                rst <= 1'b1;
                cnt = cnt + 1;
            end
            else begin
                rst <= 1'b0;
            end
            #1 clk <= 1'b0;
            #1 clk <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (!rst) begin
            
            // Caluclate simulation time
            current_time_ns = current_time_ns + NS_PER_CLK;
            
            // Put values on input whenever the timestamp is smaller than simultation time
            if (timestamp_reg * 1000 < current_time_ns) begin
                is_valid <= 1;
                timestamp_reg <= timestamps.pop_front();
                x_coord_reg   <= x_coords.pop_front();
                y_coord_reg   <= y_coords.pop_front();
                polarity_reg  <= polarities.pop_front();                
            end
            else begin
                 is_valid <= 0;
            end

            timestamp <= timestamp_reg;
            x_coord <= x_coord_reg;
            y_coord <= y_coord_reg;
            polarity <= polarity_reg;

            // Finish simulation after 50.1 ms
            if (out_valid == 1'b1) begin
                $display("%b", out_data);
                //$finish;
            end
        end
    end

    top_simon i_dut (
        .clk_i        ( clk         ),
        .rst_i      ( rst         ),
        .timestamp_i  ( timestamp   ),
        .x_coord_i    ( x_coord     ),
        .y_coord_i    ( y_coord     ),
        .polarity_i   ( polarity    ),
        .is_valid_i   ( is_valid    )
        //.out_data_o   ( out_data    ),
        //.out_valid_o  ( out_valid   )
    );


endmodule
