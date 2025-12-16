`timescale 1ns/1ps
module tb_output_accumulator_classifier;
    // Parameters
    localparam int NET_WIDTH = 8000;
    localparam int NUM_CLASSES = 10;
    localparam int NET_TO_OUT_DELAY = 2;
    localparam int MOVING_AVERAGE_DIV = 4;
    localparam int MOVING_AVERAGE_ACCUM_WIDTH = 16;

    // DUT signals
    logic clk_i = 0;
    logic reset_i = 0;
    logic [NET_WIDTH-1:0] net_i;
    logic inp_valid_i = 0;
    logic [NUM_CLASSES-1:0] class_out_o;
    logic out_valid_o;

    // Instantiate DUT
    output_accumulator_classifier #(
        .NET_WIDTH(NET_WIDTH),
        .NUM_CLASSES(NUM_CLASSES),
        .NET_TO_OUT_DELAY(NET_TO_OUT_DELAY),
        .MOVING_AVERAGE_DIV(MOVING_AVERAGE_DIV),
        .MOVING_AVERAGE_ACCUM_WIDTH(MOVING_AVERAGE_ACCUM_WIDTH)
    ) i_dut (
        .clk_i(clk_i),
        .reset_i(reset_i),
        .net_i(net_i),
        .inp_valid_i(inp_valid_i),
        .class_out_o(class_out_o),
        .out_valid_o(out_valid_o)
    );

    // Clock generation
    always #5 clk_i = ~clk_i;

    localparam N_SAMPLES = 5;

    initial begin
        // Initialize inputs
        net_i = 0;
        inp_valid_i = 0;
        reset_i = 1;
        #20;
        reset_i = 0;
        // Wait a few cycles
        #30;

        for (int i = 0; i < N_SAMPLES; i++) begin
            // Generate random net_i using std::randomize
            if (!std::randomize(net_i)) begin
                $fatal("Randomization of net_i failed");
            end
            // Wait a few cycles
            #10;
            inp_valid_i = 1;
            #10;
            inp_valid_i = 0;
            @(posedge out_valid_o);
            #10;
        end

        // Apply a few zeroes...
        for (int i = 0; i < N_SAMPLES; i++) begin
            // Generate random net_i using std::randomize
            net_i = '0; // Set all bits to zero
            // Wait a few cycles
            #10;
            inp_valid_i = 1;
            #10;
            inp_valid_i = 0;
            @(posedge out_valid_o);
            #10;
        end

        $finish;
    end
endmodule
