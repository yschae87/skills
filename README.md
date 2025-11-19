# NVIDIA Aerial cuBB - Claude Code Skills

This repository contains specialized Claude Code skills for NVIDIA Aerial cuBB (cuPHY-CP) development, providing expert knowledge on 5G PHY layer processing, O-RAN fronthaul, and FAPI interfaces.

## Overview

These skills enhance Claude Code's capabilities for developing and debugging NVIDIA Aerial SDK applications, with deep expertise in:

- **O-RAN Fronthaul**: U-Plane packet structure, compression, I/Q scaling
- **PRACH Processing**: Random access channel detection and MAC interface
- **Physical Layer**: Signal processing pipelines and GPU optimization
- **FAPI Interfaces**: 5G FAPI message formatting and PHY-MAC communication

## Available Skills

### 1. oran-uplane
**File**: `.claude/skills/oran-uplane.md`

Expert on O-RAN U-Plane packet structure, I/Q level setting, BFP compression, and beta calculation.

**Topics Covered**:
- O-RAN U-Plane packet hierarchy (Ethernet → VLAN → eCPRI → Section → PRB data)
- Beta calculation formula and parameter impacts
- BFP compression implementation
- fs_offset_dl, exponent_dl, ref_dl parameter tuning
- I/Q amplitude scaling and effective bit depth

**Use Cases**:
- Debugging fronthaul packet generation
- Tuning I/Q signal levels
- Understanding compression artifacts
- Optimizing dynamic range utilization

### 2. fs-offset
**File**: `.claude/skills/fs-offset.md`

Specialist in fs_offset_dl calculation for achieving target effective bit depths in BFP-compressed signals.

**Topics Covered**:
- Effective bit depth calculation from fs_offset_dl
- Relationship between fs_offset_dl, ref_dl, and beta_dl
- Numerical examples for different configurations
- Trade-offs between headroom and quantization noise

**Use Cases**:
- Setting specific effective bit depths (e.g., 14 bits)
- Maintaining signal quality across power levels
- Compensating for ref_dl changes

### 3. pdsch-iq-scaling
**File**: `.claude/skills/pdsch-iq-scaling.md`

Expert on PDSCH signal processing pipeline from constellation mapping through beamforming to O-RAN transmission.

**Topics Covered**:
- Complete PDSCH IQ scaling pipeline
- Beamforming weight normalization
- Beta scaling application
- BFP compression effects
- End-to-end signal level tracking

**Use Cases**:
- Understanding PDSCH signal flow
- Debugging beamforming issues
- Analyzing compression impacts
- Power control optimization

### 4. ref-dl
**File**: `.claude/skills/ref-dl-impact.md`

Specialist in ref_dl parameter impact on downlink signal processing and I/Q levels.

**Topics Covered**:
- ref_dl influence on beta_dl and signal amplitude
- Interaction with fs_offset_dl
- Power domain vs amplitude domain conversions
- Practical configuration strategies

**Use Cases**:
- Adjusting downlink power levels
- Understanding beta_dl behavior
- Coordinating fs_offset_dl and ref_dl settings

### 5. prach-processing (NEW)
**File**: `.claude/skills/prach-processing.md`

Expert on PRACH uplink processing chain from O-RAN U-Plane reception through GPU detection to FAPI RACH indication.

**Topics Covered**:
- Complete PRACH processing pipeline (O-RU → Fronthaul → GPU → FAPI → MAC)
- PhyPrachAggr class and cuPHY PRACH RX API
- GPU-accelerated preamble detection algorithms
- FAPI RACH indication message construction
- Timing advance and power estimation conversions
- PRACH configuration (root sequence, ZCZ, formats)
- Performance optimization and debugging

**Use Cases**:
- Understanding PRACH uplink flow
- Debugging random access procedures
- Tuning detection thresholds
- Analyzing timing advance calculations
- Optimizing GPU processing performance
- Troubleshooting FAPI message formatting

## Documentation

Additional technical documentation:

- **PRACH_PROCESSING_CHAIN.md**: Comprehensive technical documentation of the PRACH uplink processing chain, including architecture diagrams, API reference, performance metrics, and configuration details.

## Installation

### Prerequisites
- [Claude Code](https://claude.com/claude-code) installed
- NVIDIA Aerial SDK development environment

### Setup

1. Clone this repository to your Aerial development workspace:
```bash
cd /path/to/your/aerial/workspace
git clone <repository-url> cuBB-skills
```

2. Copy skills to your Claude Code skills directory:
```bash
# Option 1: Copy to project-specific skills
cp -r cuBB-skills/.claude/skills/* .claude/skills/

# Option 2: Create symlink for automatic updates
ln -s $(pwd)/cuBB-skills/.claude/skills ~/.claude/skills/aerial-cubb
```

3. Verify skills are loaded:
```bash
# In Claude Code
/skills
```

## Usage

Skills are automatically invoked by Claude Code when relevant topics are discussed. You can also explicitly invoke them:

### Example: O-RAN U-Plane Questions
```
User: How do I calculate beta_dl for 14 effective bits?
Claude: [Automatically invokes oran-uplane skill]
```

### Example: PRACH Processing Questions
```
User: Explain the PRACH uplink processing flow from O-RU to MAC
Claude: [Automatically invokes prach-processing skill]
```

### Example: Explicit Skill Invocation
```
User: /skill prach-processing
User: How is timing advance calculated and converted to FAPI format?
```

## Skill Development

### Adding New Skills

1. Create a new `.md` file in `.claude/skills/`
2. Follow the template structure:
```markdown
---
name: skill-name
description: Brief description for automatic invocation
---

# Skill Title

You are an expert on...

## Core Expertise
[Detailed knowledge areas]

## When to Use This Skill
[Trigger conditions]

## Response Guidelines
[How to format responses]
```

3. Test the skill with Claude Code
4. Commit and push changes

### Skill Template Structure

Each skill should include:
- **Name and Description**: For automatic invocation
- **Core Expertise**: Detailed technical knowledge organized by topic
- **Code Examples**: Concrete implementations with file paths
- **Formulas and Calculations**: Mathematical relationships with numerical examples
- **Use Cases**: When to apply this knowledge
- **Response Guidelines**: How to format answers

## Technical References

### NVIDIA Aerial SDK
- cuPHY API Documentation
- cuBB Integration Guide
- Aerial SDK Release Notes

### 3GPP Specifications
- TS 38.211: Physical channels and modulation
- TS 38.212: Multiplexing and channel coding
- TS 38.213: Physical layer procedures for control
- TS 38.214: Physical layer procedures for data

### O-RAN Specifications
- O-RAN.WG4.CUS.0: O-RAN Fronthaul Control, User and Synchronization Plane Specification
- O-RAN.WG4.IOT.0: O-RAN Fronthaul Interoperability Test Specification

### Small Cell Forum
- SCF-222: 5G FAPI: PHY API Specification

## File Structure

```
cuBB/
├── .claude/
│   └── skills/
│       ├── oran-uplane.md          # O-RAN U-Plane and I/Q scaling
│       ├── fs-offset.md            # fs_offset_dl calculation
│       ├── pdsch-iq-scaling.md     # PDSCH signal processing
│       ├── ref-dl-impact.md        # ref_dl parameter impact
│       └── prach-processing.md     # PRACH uplink processing (NEW)
├── PRACH_PROCESSING_CHAIN.md       # Detailed PRACH documentation
├── README.md                        # This file
└── .gitignore                      # Git ignore patterns
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-skill`)
3. Add your skill following the template structure
4. Test with Claude Code
5. Commit your changes (`git commit -m 'Add new skill: skill-name'`)
6. Push to the branch (`git push origin feature/new-skill`)
7. Open a Pull Request

### Contribution Guidelines

- **Accuracy**: Ensure all technical information is accurate and references specific file paths/line numbers
- **Examples**: Include concrete numerical examples and code snippets
- **Completeness**: Cover the topic comprehensively with formulas, use cases, and troubleshooting
- **Clarity**: Use clear explanations suitable for developers at various skill levels
- **References**: Cite relevant specifications (3GPP, O-RAN, FAPI)

## License

This project is provided for use with NVIDIA Aerial SDK development.

## Changelog

### 2025-11-19
- **Added**: `prach-processing.md` skill for PRACH uplink processing chain
- **Added**: `PRACH_PROCESSING_CHAIN.md` comprehensive technical documentation
- **Updated**: Repository structure with git initialization
- **Added**: README.md with complete documentation

### Previous
- Initial release with O-RAN U-Plane, fs-offset, PDSCH IQ scaling, and ref_dl skills

## Support

For questions or issues:
- Open an issue in this repository
- Consult NVIDIA Aerial SDK documentation
- Contact NVIDIA Aerial support team

## Acknowledgments

These skills are developed to support NVIDIA Aerial cuBB development and are based on:
- NVIDIA Aerial SDK documentation and source code
- 3GPP specifications for 5G NR
- O-RAN Alliance fronthaul specifications
- Small Cell Forum FAPI specifications
