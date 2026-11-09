// Timing wrapper: registers the inputs, as the decoder would see them inside a larger design,
// so place-and-route reports a register-to-register clock limit.
module timing_wrapper (
    input  wire       clk, rst, valid_in,
    input  wire [3:0] syndrome_z, syndrome_x,
    output wire       valid_out,
    output wire [8:0] correction_x, correction_z
);
  reg       v_r;
  reg [3:0] sz_r, sx_r;
  always @(posedge clk) begin
    v_r <= valid_in; sz_r <= syndrome_z; sx_r <= syndrome_x;
  end
  surface_code_d3_decoder dec (
    .clk(clk), .rst(rst), .valid_in(v_r), .syndrome_z(sz_r), .syndrome_x(sx_r),
    .valid_out(valid_out), .correction_x(correction_x), .correction_z(correction_z));
endmodule
