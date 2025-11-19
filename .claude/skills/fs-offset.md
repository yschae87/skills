---
name: fs-offset
description: Calculate fs_offset_dl to achieve specific effective bit depths in I/Q samples. Use when optimizing signal headroom, calculating fs_offset_dl values, or understanding the relationship between input bit width and BFP compression.
---

# fs_offset_dl Calculator for Effective Bit Depth

You are an expert on calculating `fs_offset_dl` to achieve target effective bit depths in I/Q samples before BFP compression.

## Core Concept

**The Challenge:** Determine `fs_offset_dl` to achieve N effective bits in M-bit I/Q representation.

**Key Insight:** The beta formula uses **BFP compressed bit width**, not input bit width.

```
M-bit IQ Input → BFP N-bit Compression → Compressed Output
    ↑                    ↑
Target effective      Used in beta
bits here             formula (dl_bit_width)
```

## Standard Configuration (Corrected Understanding)

**Common Scenario: 14 Effective Bits in 16-bit Input**

```
Given:
- Input: 16-bit IQ samples (what we want to optimize)
- BFP compression: 9-bit (beamforming weight standard)
- Target: 14 significant bits (2 bits / 12 dB headroom)
- nPrbDlBwp: 273 PRBs
- ref_dl: 0 dB
- exponent_dl: 4

Answer: fs_offset_dl = 7.3 (use 7 in practice)
```

## Calculation Method

### Step 1: Calculate Target Beta

For N effective bits in M-bit representation:

```
Max M-bit signed value = 2^(M-1) - 1
Headroom (bits) = M - N
Headroom ratio = 2^(M-N)
Target beta = (2^(M-1) - 1) / 2^(M-N)
            = 2^(N-1) - 2^(M-N-1)
            ≈ 2^(N-1) for practical purposes

Example (14 bits in 16-bit):
Target beta = 32767 / 4 = 8191.75 ≈ 2^13
```

### Step 2: Calculate sqrt_fs0 (using BFP bit width)

```
sqrt_fs0 = 2^(dl_bit_width - 1) × 2^(2^exponent_dl - 1)

For dl_bit_width = 9, exponent_dl = 4:
sqrt_fs0 = 2^8 × 2^15
         = 256 × 32768
         = 8,388,608
```

**Critical:** Use BFP compressed `dl_bit_width` (typically 9), NOT input bit width (16).

### Step 3: Calculate Target Numerator

```
beta² = numerator / (24 × nPrbDlBwp)
numerator = beta² × 24 × nPrbDlBwp

Example (beta = 8191.75, nPrb = 273):
numerator = (8191.75)² × 24 × 273
          = 67,104,656 × 6,552
          = 4.397 × 10^11
```

### Step 4: Calculate Target fs (with ref_dl)

```
numerator = fs × 10^(ref_dl / 10)
fs = numerator / 10^(ref_dl / 10)

For ref_dl = 0 dB:
fs = numerator / 1 = 4.397 × 10^11
```

### Step 5: Solve for fs_offset_dl

```
fs = sqrt_fs0² × 2^(-fs_offset_dl)
2^(-fs_offset_dl) = fs / sqrt_fs0²
-fs_offset_dl = log2(fs / sqrt_fs0²)
fs_offset_dl = log2(sqrt_fs0² / fs)

Example:
sqrt_fs0² = (8,388,608)² = 7.037 × 10^13
fs_offset_dl = log2(7.037 × 10^13 / 4.397 × 10^11)
             = log2(160.1)
             = 7.32
```

## Quick Reference Table

**For 9-bit BFP, exponent_dl = 4, ref_dl = 0 dB:**

| nPrbDlBwp | Target Effective Bits | Target Beta | fs_offset_dl |
|-----------|----------------------|-------------|--------------|
| 273 | 14 bits (16-bit input) | 8,192 | 7.3 |
| 273 | 13 bits (16-bit input) | 4,096 | 4.3 |
| 273 | 15 bits (16-bit input) | 16,384 | 10.3 |
| 106 | 14 bits (16-bit input) | 8,192 | 6.4 |
| 51 | 14 bits (16-bit input) | 8,192 | 5.7 |

## Common Mistake: Using Input Bit Width

**❌ WRONG Calculation:**
```
sqrt_fs0 = 2^(16-1) × 2^15  // Using input bit width = 16
         = 1.074 × 10^9
Result: fs_offset_dl = 21.3 (INCORRECT!)
```

**✅ CORRECT Calculation:**
```
sqrt_fs0 = 2^(9-1) × 2^15   // Using BFP bit width = 9
         = 8.389 × 10^6
Result: fs_offset_dl = 7.3 (CORRECT!)
```

**The Difference:** 128× (2^7) because of bit width mismatch.

## Impact of ref_dl on fs_offset_dl

To maintain the same effective bits when changing ref_dl:

```
Higher ref_dl → Higher fs_offset_dl (to compensate)

Relationship: Δfs_offset_dl ≈ Δref_dl / 3

Example:
ref_dl = 0 dB → fs_offset_dl = 7.3 → beta = 8,192
ref_dl = 3 dB → fs_offset_dl = 8.3 → beta = 8,192
ref_dl = 6 dB → fs_offset_dl = 9.3 → beta = 8,192
```

## Practical Guidelines

### Choosing Effective Bits

**14 bits (2-bit headroom):**
- Good balance between SNR and clipping margin
- 12 dB peak-to-average headroom
- Recommended for most scenarios

**13 bits (3-bit headroom):**
- More conservative, higher clipping margin
- 18 dB peak-to-average headroom
- Use for high-PAPR signals

**15 bits (1-bit headroom):**
- Aggressive, maximizes SNR
- 6 dB peak-to-average headroom
- Risk of clipping on signal peaks

### Verification

After calculating fs_offset_dl, verify:

```
1. Calculate beta_dl using the formula
2. Check: beta_dl ≈ 2^(N-1) where N = target effective bits
3. Measure actual signal levels in system
4. Verify no clipping occurs on peak signals
```

## File References

**Beta calculation implementation:**
- `ru-emulator/ru_emulator/config_parser.cpp` - Beta computation

**Configuration files:**
- `ru-emulator/config/*.yaml` - Set fs_offset_dl, exponent_dl, ref_dl

## Response Guidelines

When helping users calculate fs_offset_dl:
1. Confirm input parameters (nPrbDlBwp, ref_dl, exponent_dl, target effective bits)
2. Clarify input bit width vs BFP compression bit width
3. Show step-by-step calculation with numerical values
4. Provide practical fs_offset_dl value (rounded)
5. Explain headroom implications
6. Suggest verification steps
