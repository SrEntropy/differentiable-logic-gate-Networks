module output_accumulator_classifier #(
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

    logic[$clog2(NUM_SLIDING_WINDOW_STEPS)-1:0] sliding_window_pos_q, sliding_window_pos_d;
    
    // After inp_valid_delay is asserted, sweep the sliding window over the network output
    //    continue sliding even if inp_valid is deasserted 
    always_comb begin
        sliding_window_pos_d = sliding_window_pos_q;
        if (sliding_window_pos_q == NUM_SLIDING_WINDOW_STEPS) begin
            sliding_window_pos_d = '0;
        end else if (inp_valid_i || sliding_window_pos_q != 0) begin
            sliding_window_pos_d = sliding_window_pos_q + 1;
        end
    end

    // Argmax over the moving averages
    assign out_valid_o = (sliding_window_pos_q == NUM_SLIDING_WINDOW_STEPS);
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

    // TODO: probably the sliding window can be made even smaller. We have plently of time between 
    //          fully accumulating frames. 

    logic[MOVING_AVERAGE_ACCUM_WIDTH-1:0] current_sliding_window_zero_count;
    // Count the number of events in the sliding window 
    always_comb begin
        current_sliding_window_zero_count = 0;
        for (int i = 0; i < SLIDING_WINDOW_WIDTH; i++) begin
            if (net_i[i + sliding_window_pos_q * SLIDING_WINDOW_WIDTH]) begin
                current_sliding_window_zero_count++;
            end
        end
    end

    always_comb begin
        for (int i = 0; i < NUM_CLASSES; i++) begin
            class_moving_average_d[i] = class_moving_average_q[i];
            if (sliding_window_pos_q == i) begin
                class_moving_average_d[i] = (class_moving_average_q[i]/MOVING_AVERAGE_DIV*MOVING_AVERAGE_SCALE + current_sliding_window_zero_count);
            end
        end
    end

    always_ff @(posedge clk_i or posedge reset_i) begin
        if (reset_i) begin
            sliding_window_pos_q <= '0;
        end else begin
            sliding_window_pos_q <= sliding_window_pos_d;
        end
    end

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