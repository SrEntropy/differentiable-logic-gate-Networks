/*
 * Copyright (c) 2025 Renaldas Zioma
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module telluride_mnist (
    //input  wire [7:0] pixels_in,
    //input  wire       write_enable,
    input wire [33: 0]  timestamp,
    input wire [13 : 0] x_coord,
    input wire [13 : 0] y_coord,
    input wire          polarity,
    input wire          is_valid,
    
    output wire [7:0] pmod_a_out,
    //output wire [7:0] pmod_b_out,
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
  // List all unused inputs to prevent warnings
  wire _unused = &{rst_n, 1'b0};

  localparam INPUTS  = 64*64;
  localparam CATEGORIES = 10;
  localparam BITS_PER_CATEGORY = 800;
  localparam OUTPUTS = BITS_PER_CATEGORY * CATEGORIES;
  localparam BITS_PER_CATEGORY_SUM = $clog2(BITS_PER_CATEGORY);

  wire   [INPUTS-1:0] vector;
  wire   [INPUTS-1:0] x;
  reg   [INPUTS-1:0] x_reg;
  wire x_valid;

  top #(
        .INPUT_SIZE ( INPUTS )
  ) decode_event(
        .clk        ( clk          ),
        .reset      ( rst_n        ),
        .timestamp  ( timestamp    ),
        .x_coord    ( x_coord      ),
        .y_coord    ( y_coord      ),
        .polarity   ( polarity     ),
        .is_valid   ( is_valid     ),
        .out_data   ( vector       ),
        .out_valid  ( x_valid      )
  );

  
  wire [OUTPUTS-1:0] y;
  wire [BITS_PER_CATEGORY*CATEGORIES-1:0] out_categories;
  reg [BITS_PER_CATEGORY*CATEGORIES-1:0] reg_categories;
  wire [BITS_PER_CATEGORY*CATEGORIES-1:0] y_categories;

  always @(posedge clk) begin
    x_reg <= vector;
    reg_categories <= out_categories;
  end

  assign x = x_reg;
  assign y_categories = reg_categories;

  net net(
    .clk(clk),
    .in(x),
    .out(y),
    .categories(out_categories)
  );

  wire [BITS_PER_CATEGORY_SUM*CATEGORIES-1:0] sum_categories;
  reg [BITS_PER_CATEGORY_SUM*CATEGORIES-1:0] sum_categories_reg;
  wire [BITS_PER_CATEGORY_SUM*CATEGORIES-1:0] sum_categories_wire;

  genvar i;
  generate
    for (i = 0; i < CATEGORIES; i = i+1) begin : calc_categories
      sum_bits #(.N(BITS_PER_CATEGORY)) sum_bits(
        .y(y_categories[i*BITS_PER_CATEGORY +: BITS_PER_CATEGORY]),      
        .sum(sum_categories[i*BITS_PER_CATEGORY_SUM +: BITS_PER_CATEGORY_SUM])
      );
    end
  endgenerate

  always @(posedge clk) begin
    sum_categories_reg <= sum_categories;
  end

  assign sum_categories_wire = sum_categories_reg;

  wire [3:0] best_category_index;
  reg [3:0] best_category_index_reg;
  wire [3:0] best_category_index_wire;

  always @(posedge clk) begin
    best_category_index_reg <= best_category_index;
  end

  assign best_category_index_wire = best_category_index_reg;

  wire [7:0] best_category_value;
  arg_max_10 #(.N(BITS_PER_CATEGORY_SUM)) arg_max_categories(
    .categories(sum_categories_wire),
    .out_index(best_category_index),
    .out_value(best_category_value)
  );


  wire [6:0] display;
  seven_segment seven_segment(
    .in(best_category_index_wire[3:0]),
    .out(display)
  );

  assign pmod_a_out[6:0] = ~display;
  assign pmod_a_out[7] = 0; //~x_valid;
//  assign pmod_b_out[6:0] = best_category_value[6:0];
//  assign pmod_b_out[7]   = 0;
endmodule

module sum_bits #(
    parameter N = 16
) (
    input wire [N-1:0] y,
    output wire [$clog2(N)-1:0] sum
);
    integer i;
    reg [$clog2(N)-1:0] temp_sum;
    
    always @(*) begin
        temp_sum = 0;
        for (i = 0; i < N; i = i + 1) begin
            temp_sum = temp_sum + y[i];
        end
    end
    
    assign sum = temp_sum;
endmodule

module arg_max_10 #(
    parameter N = 8
) (
    input wire [10*N-1:0] categories,
    output reg [3:0] out_index,
    output reg [7:0] out_value
);
    // Intermediate wires for the tree comparison
    (* mem2reg *) reg [N-1:0] max_value_stage1 [4:0];  // Stage 1: Compare adjacent pairs
    (* mem2reg *) reg [3:0]   max_index_stage1 [4:0]; 

    (* mem2reg *) reg [N-1:0] max_value_stage2 [2:0];  // Stage 2: Compare reduced pairs
    (* mem2reg *) reg [3:0]   max_index_stage2 [2:0]; 

                  reg [N-1:0] max_value_stage3;        // Stage 3: Final comparison
                  reg [3:0]   max_index_stage3;

    integer i;

    always @(*) begin
        // Stage 1: Compare adjacent pairs
        for (i = 0; i < 5; i = i + 1) begin
            if (categories[(2*i)*N +: N] > categories[(2*i+1)*N +: N]) begin
                max_value_stage1[i] = categories[(2*i)*N +: N];
                max_index_stage1[i] = 2*i;
            end else begin
                max_value_stage1[i] = categories[(2*i+1)*N +: N];
                max_index_stage1[i] = 2*i+1;
            end
        end

        // Stage 2: Compare reduced pairs
        for (i = 0; i < 2; i = i + 1) begin
            if (max_value_stage1[2*i] > max_value_stage1[2*i+1]) begin
                max_value_stage2[i] = max_value_stage1[2*i];
                max_index_stage2[i] = max_index_stage1[2*i];
            end else begin
                max_value_stage2[i] = max_value_stage1[2*i+1];
                max_index_stage2[i] = max_index_stage1[2*i+1];
            end
        end
        // Handle the last element (if odd number of inputs)
        max_value_stage2[2] = max_value_stage1[4];
        max_index_stage2[2] = max_index_stage1[4];

        // Stage 3: Final comparison
        if (max_value_stage2[0] > max_value_stage2[1]) begin
            if (max_value_stage2[0] > max_value_stage2[2]) begin
                max_value_stage3 = max_value_stage2[0];
                max_index_stage3 = max_index_stage2[0];
            end else begin
                max_value_stage3 = max_value_stage2[2];
                max_index_stage3 = max_index_stage2[2];
            end
        end else begin
            if (max_value_stage2[1] > max_value_stage2[2]) begin
                max_value_stage3 = max_value_stage2[1];
                max_index_stage3 = max_index_stage2[1];
            end else begin
                max_value_stage3 = max_value_stage2[2];
                max_index_stage3 = max_index_stage2[2];
            end
        end

        // Assign final max index
        out_index = max_index_stage3;
        out_value = max_value_stage3;
    end
endmodule


//          1
//         ---
//        |   |
//      6 | 7 | 2
//         ---
//        |   |
//      5 | 4 | 3
//         ---

module seven_segment (
    input  wire [3:0] in,
    output reg  [6:0] out
);
    always @(*) begin
        case(in)
            //          .7654321
            0:  out = 7'b0111111;
            1:  out = 7'b0000110;
            2:  out = 7'b1011011;
            3:  out = 7'b1001111;
            4:  out = 7'b1100110;
            5:  out = 7'b1101101;
            6:  out = 7'b1111100;
            7:  out = 7'b0000111;
            8:  out = 7'b1111111;
            9:  out = 7'b1100111;
            default:
                out = 7'b0000000;
        endcase
    end
endmodule

