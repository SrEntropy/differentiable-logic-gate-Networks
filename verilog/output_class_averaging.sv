module output_class_averaging #(
    parameter int NUM_CLASSES = 10,
    parameter int NUM_AVERAGING_STEPS = 4
)( 
	input logic	clk_i,
	input logic	reset_i,
    input logic inp_valid_i,  // Assume this comes delayed from the top. 
    output logic [$clog2(NUM_CLASSES)-1:0] class_MA_i, // 0:9 instead of 1:10
    
    output logic [$clog2(NUM_CLASSES)-1:0] class_avg_o // 0:9 instead of 1:10
    
);

    // Shift register to store the last NUM_AVERAGING_STEPS classes
    logic [$clog2(NUM_CLASSES)-1:0] class_shift_reg [NUM_AVERAGING_STEPS-1:0];
    
    always_ff @(posedge clk_i or posedge reset_i) begin
        if (reset_i) begin
            for (int i = 0; i < NUM_AVERAGING_STEPS; i++) begin
                class_shift_reg[i] <= '0;
            end
        end else if (inp_valid_i) begin
            // Shift the classes in the register
            for (int i = NUM_AVERAGING_STEPS-1; i > 0; i--) begin
                class_shift_reg[i] <= class_shift_reg[i-1];
            end
            // Insert the new class at the front
            class_shift_reg[0] <= class_MA_i;
        end
    end

    // Find the most common class in the shift register
    logic [$clog2(NUM_CLASSES)-1:0] class_count [NUM_CLASSES-1:0];
    always_comb begin
        for (int i = 0; i < NUM_CLASSES; i++) begin
            class_count[i] = '0;
            for (int j = 0; j < NUM_AVERAGING_STEPS; j++) begin
                if (class_shift_reg[j] == i) 
                    class_count[i]++;
            end
        end
     end 
    logic [$clog2(NUM_CLASSES)-1:0] max_class;
    always_comb begin
        max_class = '0;
        for (int i = 0; i < NUM_CLASSES; i++) begin
            if (class_count[i] > class_count[max_class]) begin
                max_class = i;
            end
        end
    end

    assign class_avg_o = max_class;
endmodule