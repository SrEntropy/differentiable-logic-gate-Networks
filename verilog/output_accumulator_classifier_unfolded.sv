module output_accumulator_classifier_unfolded #(
    parameter int NET_WIDTH = 8000,
    parameter int NUM_CLASSES = 10,
    parameter int NET_TO_OUT_DELAY = 2,
    parameter int MOVING_AVERAGE_DIV = 4,
    parameter int MOVING_AVERAGE_SCALE = 1,
    parameter int MOVING_AVERAGE_ACCUM_WIDTH = 16
)( 
	input logic	clk_i,
	input logic	reset_i,
    input logic [NET_WIDTH-1:0] net_i,
    input logic inp_valid_i,  // Assume this comes delayed from the top. 

    output logic [$clog2(NUM_CLASSES)-1:0] class_out_o, // 0:9 instead of 1:10
    output logic out_valid_o
);

    localparam NUM_SLIDING_WINDOW_STEPS = NUM_CLASSES;
    localparam SLIDING_WINDOW_WIDTH = NET_WIDTH / NUM_SLIDING_WINDOW_STEPS;

    logic [MOVING_AVERAGE_ACCUM_WIDTH-1:0] class_moving_average_q [NUM_CLASSES-1:0];
    logic [MOVING_AVERAGE_ACCUM_WIDTH-1:0] class_moving_average_d [NUM_CLASSES-1:0];
    logic[MOVING_AVERAGE_ACCUM_WIDTH-1:0] class_popcount [NUM_CLASSES-1:0];

    // After inp_valid_delay is asserted, sweep the sliding window over the network output
    //    continue sliding even if inp_valid is deasserted 

    // Argmax over the moving averages
    assign out_valid_o = inp_valid_i; 
    logic [MOVING_AVERAGE_ACCUM_WIDTH-1:0] current_max_temp;
    always_comb begin
        current_max_temp = 0;
        class_out_o = '0; 
        for (int i = 0; i < NUM_CLASSES; i++) begin
            if (class_moving_average_q[i] > current_max_temp) begin
                class_out_o = i;
                current_max_temp = class_moving_average_q[i];
            end
        end
    end

    // Generate class-wise popcount and moving average logic
    genvar class_idx;
    generate
        for (class_idx = 0; class_idx < NUM_CLASSES; class_idx++) begin : gen_class_logic
            always_comb begin
                // Default assignment
                class_popcount[class_idx] = 0;
                class_moving_average_d[class_idx] = class_moving_average_q[class_idx];

                for (int j = 0; j < SLIDING_WINDOW_WIDTH; j++) begin
                    if (net_i[j + class_idx * SLIDING_WINDOW_WIDTH]) begin
                        class_popcount[class_idx]++;
                    end
                end

                // Compute moving average update
                if (inp_valid_i)
                    class_moving_average_d[class_idx] = (class_moving_average_q[class_idx] / MOVING_AVERAGE_DIV * MOVING_AVERAGE_SCALE) + class_popcount[class_idx];
            end
        end
    endgenerate    

    always_ff @(posedge clk_i or posedge reset_i) begin
        if (reset_i) begin
            for (int i = 0; i < NUM_CLASSES; i++) begin
                class_moving_average_q[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_CLASSES; i++) begin
                class_moving_average_q[i] = class_moving_average_d[i];
            end
        end
    end


endmodule