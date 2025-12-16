module tellurdie_wrapper (

    input wire [33: 0]  timestamp_i,
    input wire [13 : 0] x_coord_i,
    input wire [13 : 0] y_coord_i,
    input wire          polarity_i,
    input wire          is_valid_i,
    
    output wire [7:0] pmod_a_out_o,

    input  wire       clk_i,
    input  wire       rst_i,

    output wire                  DEBUG_preprocessing_data_valid_o,
    output wire [$clog2(10)-1:0] DEBUG_class_out_MA_o,
    output wire                  DEBUG_out_valid_MA_o
);

    top_simon top_module (
        .clk_i(clk_i),
        .reset_i(rst),
        .timestamp_i(timestamp_i),
        .x_coord_i(x_coord_i),
        .y_coord_i(y_coord_i),
        .polarity_i(polarity_i),
        .is_valid_i(is_valid_i),
        .pmod_a_out_o(pmod_a_out_o),
        .DEBUG_preprocessing_data_valid_o(DEBUG_preprocessing_data_valid_o),
        .DEBUG_class_out_MA_o(DEBUG_class_out_MA_o),
        .DEBUG_out_valid_MA_o(DEBUG_out_valid_MA_o)
    );

endmodule