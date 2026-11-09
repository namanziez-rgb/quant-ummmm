// Exhaustive testbench: drives all 256 syndrome pairs, one per clock cycle,
// and checks each correction one cycle later against expected.hex from Python.
`timescale 1ns/1ps
module tb_surface_code_d3_decoder;
  reg        clk = 0, rst = 1, valid_in = 0;
  reg  [3:0] syndrome_z = 0, syndrome_x = 0;
  wire       valid_out;
  wire [8:0] correction_x, correction_z;
  reg  [17:0] expected [0:255];
  integer i, errors = 0, checked = 0;

  surface_code_d3_decoder dut (
    .clk(clk), .rst(rst), .valid_in(valid_in),
    .syndrome_z(syndrome_z), .syndrome_x(syndrome_x),
    .valid_out(valid_out), .correction_x(correction_x), .correction_z(correction_z));

  always #5 clk = ~clk;   // 100 MHz in simulation

  initial begin
    $readmemh("expected.hex", expected);
    @(posedge clk); @(posedge clk); rst <= 0;
    for (i = 0; i <= 256; i = i + 1) begin
      // check the result for the syndrome applied one cycle earlier
      if (i > 0) begin
        if (!valid_out || {correction_z, correction_x} !== expected[i - 1]) begin
          errors = errors + 1;
          $display("MISMATCH idx=%0d got z=%b x=%b want %h", i - 1, correction_z, correction_x, expected[i - 1]);
        end
        checked = checked + 1;
      end
      if (i < 256) begin
        valid_in   <= 1;
        syndrome_z <= i[3:0];
        syndrome_x <= i[7:4];
      end else valid_in <= 0;
      @(posedge clk); #1;
    end
    if (errors == 0) $display("PASS: %0d of %0d syndrome pairs match the Python decoder", checked, checked);
    else             $display("FAIL: %0d mismatches out of %0d", errors, checked);
    $finish;
  end
endmodule
