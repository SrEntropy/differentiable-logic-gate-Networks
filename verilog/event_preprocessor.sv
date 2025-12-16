module event_preprocessor #(
    parameter int INPUT_WIDTH = 64,
    parameter int SPATIAL_DOWNSAMPLING_FACTOR = 2, // Downsampling factor for x and y coordinates
    parameter int EVENT_INTEGRATION_THRESHOLD = 100 // As soon as 100 events are received, output is generated
)( 
	input logic		                clk_i,
	input logic			            reset_i,
	input logic [33: 0]             timestamp_i, // Unused 
	input logic [13 : 0]            x_coord_i,
	input logic [13 : 0]            y_coord_i,
	input logic			            polarity_i,
	input logic                     is_valid_i,

    output logic [(INPUT_WIDTH/SPATIAL_DOWNSAMPLING_FACTOR)**2-1 : 0] out_data_o,
    output logic                    out_valid_o

);  

    localparam EVENT_BITWIDTH = 14;
    localparam TIMESTAMP_BITWIDTH = 34;
    localparam INPUT_SIZE = INPUT_WIDTH**2;
    localparam OUTPUT_WIDTH = INPUT_WIDTH / SPATIAL_DOWNSAMPLING_FACTOR;
    localparam OUTPUT_SIZE = OUTPUT_WIDTH**2;

    logic [OUTPUT_SIZE-1:0] out_vector_d, out_vector_q;
    logic [$clog2(OUTPUT_SIZE):0] x_coord_downsampled, y_coord_downsampled, addr_downsampled;
    logic [$clog2(EVENT_INTEGRATION_THRESHOLD):0] event_counter_d, event_counter_q;

    logic event_counter_threshold_reached;
    assign event_counter_threshold_reached = (event_counter_q == EVENT_INTEGRATION_THRESHOLD);

    assign out_data_o = out_vector_q;
    assign out_valid_o = event_counter_threshold_reached;

    always_comb begin
        out_vector_d = out_vector_q;
        event_counter_d = event_counter_q;
        x_coord_downsampled = '0;
        y_coord_downsampled = '0;
        addr_downsampled = '0;

        // Only integrate positive events for now. 
        if (event_counter_threshold_reached) begin
            out_vector_d = '0;
            out_valid_o = 1'b1;
            event_counter_d = '0;

        // Only accumulate positive events for now.
        end else if (is_valid_i && polarity_i) begin
            x_coord_downsampled = x_coord_i / SPATIAL_DOWNSAMPLING_FACTOR;
            y_coord_downsampled = y_coord_i / SPATIAL_DOWNSAMPLING_FACTOR;
            addr_downsampled = OUTPUT_WIDTH * y_coord_downsampled + x_coord_downsampled;
            
            // Only increment the threshold counter if event is not already set. 
            if (!out_vector_q[addr_downsampled]) begin
                out_vector_d[addr_downsampled] = 1'b1;
                event_counter_d = event_counter_q + 1;
            end
        end
    end

    always_ff @(posedge clk_i or posedge reset_i) begin
        if (reset_i) begin
            out_vector_q <= '0;
            event_counter_q <= '0;
        end else begin
            out_vector_q <= out_vector_d;
            event_counter_q <= event_counter_d;
        end
    end

endmodule