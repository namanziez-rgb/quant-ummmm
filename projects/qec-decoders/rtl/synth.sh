#!/bin/sh
# Synthesize for a Lattice iCE40 UP5K and report logic usage and the clock limit.
# Needs: pip install yowasp-yosys yowasp-nextpnr-ice40
set -e
yowasp-yosys -q -p "read_verilog surface_code_d3_decoder.v timing_wrapper.v; \
  synth_ice40 -top timing_wrapper -json build.json; tee -o synth_stats.txt stat"
yowasp-nextpnr-ice40 --up5k --package sg48 --json build.json --freq 100 \
  --report pnr_report.json 2> pnr.log
grep -E "SB_LUT4|SB_DFF" synth_stats.txt
grep "Max frequency" pnr.log | tail -1
