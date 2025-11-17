# NVIDIA Aerial cuPHY Skills

This directory contains Claude Code skills for working with NVIDIA Aerial cuPHY and cuPHY-CP codebases.

## Skills

### oran-uplane.md
Expert on O-RAN U-Plane packet structure, I/Q level setting, BFP compression, and beta calculation in cuPHY-CP. Use when discussing:
- O-RAN fronthaul packet generation
- U-Plane packet structure and headers
- Beta calculation and I/Q scaling
- BFP compression implementation
- fs_offset_dl, exponent_dl, ref_dl parameters

### fs-offset.md
Calculate fs_offset_dl to achieve specific effective bit depths in I/Q samples. Use when:
- Optimizing signal headroom
- Calculating fs_offset_dl values
- Understanding the relationship between input bit width and BFP compression
- Working with 14-bit effective resolution in 16-bit samples

### pdsch-iq-scaling.md
Expert on PDSCH signal processing pipeline including QAM modulation, beamforming, and BFP compression. Use when discussing:
- PDSCH transmission processing
- QAM modulation and constellation mapping
- Beamforming coefficients and MMSE algorithm
- Multi-stage scaling factors (beta_qam, lambda, beta)
- Power budget analysis

### ref-dl-impact.md
Quick reference for ref_dl parameter impact on beta_dl amplitude scaling. Use when:
- Understanding ref_dl parameter effects
- Calculating amplitude scaling relationships
- Maintaining constant effective bits with different ref_dl settings
- Converting between power and amplitude domain dB values

## Context

These skills support development and analysis of:
- **cuPHY**: NVIDIA's CUDA-based Physical Layer SDK for 5G NR
- **cuPHY-CP**: Complete 5G L1 PHY Control Plane with O-RAN fronthaul support
- **O-RAN Fronthaul**: Implementation of O-RAN.WG4.CUS.0-v05.00 specification

## Usage

These skills are designed for use with Claude Code. Place them in your `.claude/skills/` directory to enable specialized assistance with Aerial SDK development.

## Related Documentation

The skills are distilled from comprehensive analysis documents covering:
- O-RAN U-Plane packet generation and BFP compression
- I/Q level setting mechanisms and beta calculations
- PDSCH modulation and scaling pipelines
- Parameter optimization for effective bit depth

## License

These skills are provided for NVIDIA Aerial SDK development.
