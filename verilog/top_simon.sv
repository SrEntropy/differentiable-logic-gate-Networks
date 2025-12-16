module top_simon (
    //input  logic [7:0] pixels_in,
    //input  logic       write_enable,
    input logic [33: 0]  timestamp_i,
    input logic [13 : 0] x_coord_i,
    input logic [13 : 0] y_coord_i,
    input logic          polarity_i,
    input logic          is_valid_i,
    
    output logic [7:0] pmod_a_out_o,

    input  logic       clk_i,
    input  logic       rst_i,

    output logic                           DEBUG_preprocessing_data_valid_o,
    output logic [$clog2(NUM_CLASSES)-1:0] DEBUG_class_out_MA_o,
    output logic                           DEBUG_out_valid_MA_o
);

    localparam INPUT_WIDTH = 64;
    localparam SPATIAL_DOWNSAMPLING_FACTOR = 1; // Keep pow2
    localparam EVENT_INTEGRATION_THRESHOLD = 1; // Set low for testing.. Change later!
     
    localparam NUM_CLASSES = 10;
    localparam NET_TO_OUT_DELAY = 2;
    localparam MOVING_AVERAGE_DIV = 4; // Keep pow2
    localparam MOVING_AVERAGE_ACCUM_WIDTH = 16; // Not sure what a reasonable value is here. Trial and error :) 

    localparam NET_INPUT_SIZE = (INPUT_WIDTH / SPATIAL_DOWNSAMPLING_FACTOR)**2;
    localparam NET_OUTPUT_SIZE = 8000;

    logic [NET_INPUT_SIZE-1:0] preprocessed_data;
    logic [NET_OUTPUT_SIZE-1:0] net_output;

    logic out_valid_preprocessing, out_valid_MA; // Should gate net inputs somehwere... or just let it run and ignore output? 
    logic [$clog2(NUM_CLASSES)-1:0] class_out_MA;
    logic [$clog2(NUM_CLASSES)-1:0] class_out_avg;

    logic[6:0] display; 

    // Debug signals for ILA
    assign DEBUG_preprocessing_data_valid_o = out_valid_preprocessing;
    assign DEBUG_class_out_MA_o = class_out_MA;
    assign DEBUG_out_valid_MA_o = out_valid_MA;

    logic rst;
    assign rst = rst_i; // Change depending on if top is active low or high reset. 
    //assign rst = ~rst_ni;

    // Pre-process data with spatial downsampling and event count thresholding
    event_preprocessor #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .SPATIAL_DOWNSAMPLING_FACTOR(SPATIAL_DOWNSAMPLING_FACTOR),
        .EVENT_INTEGRATION_THRESHOLD(EVENT_INTEGRATION_THRESHOLD)
    ) i_event_preprocessor (
        .clk_i(clk_i),
        .reset_i(rst),
        .timestamp_i(timestamp_i),
        .x_coord_i(x_coord_i),
        .y_coord_i(y_coord_i),
        .polarity_i(polarity_i),
        .is_valid_i(is_valid_i),
        .out_data_o(preprocessed_data),
        .out_valid_o(out_valid_preprocessing)
    );

    // output accumulator and moving average classifier. Outputs max of the MA group

    // To account for the net delay. 
    logic net_done_processing;
    variable_delay #(
        .N_DELAY_CYCLES(NET_TO_OUT_DELAY),
        .SIGNAL_WIDTH(1)
    ) i_variable_delay (
        .clk_i   (clk_i),       // Clock input
        .rst_i   (rst),       // Asynchronous reset input
        .data_i  (out_valid_preprocessing),   // Input data signal
        .data_o  (net_done_processing)   // Output after delay
    );

    // Net is currently 4096 input, 8000 output.
    // diff net instantiation
    net i_diff_logic_network(
        .clk(clk_i), // Maybe just add valid signal to the network? 
        .in(preprocessed_data),
        .out(net_output)
    );


    output_accumulator_classifier_unfolded #(
        .NET_WIDTH(NET_OUTPUT_SIZE),
        .NUM_CLASSES(NUM_CLASSES),
        .NET_TO_OUT_DELAY(NET_TO_OUT_DELAY),
        .MOVING_AVERAGE_DIV(MOVING_AVERAGE_DIV),
        .MOVING_AVERAGE_ACCUM_WIDTH(MOVING_AVERAGE_ACCUM_WIDTH)
    ) i_output_accumulator_classifier (
        .clk_i(clk_i),
        .reset_i(rst),
        .net_i(net_output),
        .inp_valid_i(net_done_processing),
        .class_out_o(class_out_MA),
        .out_valid_o(out_valid_MA)
    );

    output_class_averaging #(
        .NUM_CLASSES(NUM_CLASSES),
        .NUM_AVERAGING_STEPS(10)
    ) i_output_class_averaging (
        .clk_i(clk_i),
        .reset_i(rst),
        .inp_valid_i(out_valid_MA), // Use the valid signal from the classifier
        .class_MA_i(class_out_MA),
        .class_avg_o(class_out_avg) // Output the averaged class
    );
    


    // Seven segment display decoder

    //          1
    //         ---
    //        |   |
    //      6 | 7 | 2
    //         ---
    //        |   |
    //      5 | 4 | 3
    //         ---

    always_comb begin
        display = 7'b0; // Default to off
        unique case (class_out_avg)
            0: display = 7'b0111111; // 0
            1: display = 7'b0000110; // 1
            2: display = 7'b1011011; // 2
            3: display = 7'b1001111; // 3
            4: display = 7'b1100110; // 4
            5: display = 7'b1101101; // 5
            6: display = 7'b1111100; // 6
            7: display = 7'b0000111; // 7
            8: display = 7'b1111111; // 8
            9: display = 7'b1100111; // 9
            default: display = 7'b0000000;
        endcase
    end
    
    // I dont know why this is done like this ? Ray pls comment :) 
    assign pmod_a_out_o[6:0] = display;
    assign pmod_a_out_o[7] = 1'b0; 
endmodule