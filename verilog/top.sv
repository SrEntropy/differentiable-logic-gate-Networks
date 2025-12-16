`timescale 1ns / 1ps

module top #(
    parameter int X_SIZE = 64,
	parameter int INPUT_SIZE	 = 64*64
)( 
	input logic		                clk,
	input logic			            reset,
	input logic [33: 0]             timestamp,
	input logic [13 : 0]            x_coord,
	input logic [13 : 0]            y_coord,
	input logic			            polarity,
	input logic                     is_valid,

    output logic [INPUT_SIZE-1 : 0] out_data,
    output logic                    out_valid

);
    localparam EVENT_INTEGRATION_TIMEWINDOW = 1000;

    logic [INPUT_SIZE-1:0] generate_data = '0;
    logic [$clog2(INPUT_SIZE):0] addr;
    logic clear = 0;
    logic [33: 0] time_window_thr = EVENT_INTEGRATION_TIMEWINDOW;

    assign addr = (X_SIZE*(y_coord/2)) + (x_coord/2);

    always @(posedge clk) begin
        if (reset) begin
            generate_data <= '0;
            clear <= '0;
            time_window_thr <= EVENT_INTEGRATION_TIMEWINDOW;
            out_valid <= '0;
            out_data <= '0;
        end
        else begin
            out_valid <= '0;
            if (clear) begin
                clear <= '0;
                generate_data <= '0;
            end
            if(is_valid) begin
                if (timestamp > time_window_thr) begin
                    out_valid <= 1;
                    out_data <= generate_data;
                    time_window_thr <= time_window_thr + EVENT_INTEGRATION_TIMEWINDOW;
                    clear <= '1;
                end
                else begin
                    generate_data[addr] <= 1;
                end
            end
        end
    end



endmodule : top
