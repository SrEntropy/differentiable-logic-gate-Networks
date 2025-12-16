module variable_delay #(
    parameter int N_DELAY_CYCLES = 4,               // Delay in clock cycles
    parameter int SIGNAL_WIDTH = 1            // Bit-width of the signal
)(
    input  logic             clk_i,
    input  logic             rst_i,
    input  logic [SIGNAL_WIDTH-1:0] data_i,
    output logic [SIGNAL_WIDTH-1:0] data_o
);

    // Delay shift register
    logic [SIGNAL_WIDTH-1:0] shift_reg [0:N_DELAY_CYCLES-1];

    // Sequential logic for shift register
    integer i;
    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            for (i = 0; i < N_DELAY_CYCLES; i++) begin
                shift_reg[i] <= '0;
            end
        end else begin
            shift_reg[0] <= data_i;
            for (i = 1; i < N_DELAY_CYCLES; i++) begin
                shift_reg[i] <= shift_reg[i-1];
            end
        end
    end

    assign data_o = shift_reg[N_DELAY_CYCLES-1];

endmodule
