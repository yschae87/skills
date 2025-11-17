---
name: pdsch-iq-scaling
description: Expert on PDSCH signal processing pipeline including QAM modulation, beamforming, and BFP compression with scaling factors. Use when discussing PDSCH TX, modulation mapping, beamforming coefficients, or multi-stage scaling in cuPHY.
---

# PDSCH IQ Modulation and Scaling Expert

You are an expert on the complete PDSCH (Physical Downlink Shared Channel) signal processing pipeline in NVIDIA cuPHY, from QAM modulation through beamforming to BFP compression.

## Complete Pipeline

```
Rate-Matched Bits → QAM Modulation (beta_qam) →
Beamforming (lambda, normalization) → BFP Compression (beta) →
Compressed Output
```

## 1. QAM Modulation Stage

**File:** `src/cuphy/modulation_mapper/modulation_mapper.cu`

### Constellation Normalization

Per 3GPP TS 38.211 Section 5.1:

| Modulation | Normalization | Value | Power |
|------------|---------------|-------|-------|
| QPSK | 1/√2 | 0.7071 | 1.0 |
| 16-QAM | 1/√10 | 0.3162 | 1.0 |
| 64-QAM | 1/√42 | 0.1543 | 1.0 |
| 256-QAM | 1/√170 | 0.0767 | 1.0 |

### Beta_qam Parameter

**Purpose:** Additional amplitude scaling beyond standard normalization

```c
struct PdschDmrsParams {
    float beta_qam;     // QAM amplitude scaling factor
    uint8_t num_layers;
    uint16_t num_Rbs;
    // ...
};
```

**Typical Values:**
- Standard: `beta_qam = 1.0` (no additional scaling)
- Power boosting: `beta_qam > 1.0` (increase symbol power)
- Power reduction: `beta_qam < 1.0` (decrease symbol power)

### QPSK Implementation

```cuda
__device__ void modulation_QPSK(...) {
    float scale = 0.707106781186547f * params->beta_qam;  // (1/√2) × beta_qam

    __half2 symbol;
    symbol.x = (bit0 == 0) ? scale : -scale;  // I component
    symbol.y = (bit1 == 0) ? scale : -scale;  // Q component
}
```

**Example:**
```
Input bits: [0, 0]
Beta_qam: 1.0
I = +0.7071, Q = +0.7071
Power: |I|² + |Q|² = 0.5 + 0.5 = 1.0
```

### Constellation Tables

**16-QAM** (`modulation_mapper.cu:24-32`):

```cuda
__device__ __constant__ float rev_qam_16[4] = {
    0.316227766,     // +1/√10
    -0.316227766,    // -1/√10
    0.948683298,     // +3/√10
    -0.948683298     // -3/√10
};
```

**Gray Coding:** Bits {0,2} → I, Bits {1,3} → Q

### Higher-Order Modulation

**64-QAM:**
- 8 constellation points per dimension: ±1/√42, ±3/√42, ±5/√42, ±7/√42
- Gray coding with bits {0,2,4} → I, {1,3,5} → Q

**256-QAM:**
- 16 constellation points per dimension
- Bits {0,2,4,6} → I, {1,3,5,7} → Q

### Output Format

**Data type:** `__half2` (FP16 complex)

**Layout modes:**
1. **Flat mode** (`num_Rbs = 0`): Contiguous symbol array
2. **Spatial mode** (`num_Rbs > 0`): 3D tensor [REs × Symbols × Layers]

## 2. Beamforming Stage

**File:** `src/cuphy/bfc/bfc.cu`

### MMSE Beamforming

**Algorithm:**

```
Given:
- H: Channel matrix (N_LAYERS × N_BS_ANTS)
- λ (lambda): Regularization constant

Compute:
1. Gram matrix: G = H × H^H + λI
2. LU factorization: G = L × U
3. MMSE coefficients: C = H^H × G^(-1)
```

Where C is the beamforming coefficient matrix (N_BS_ANTS × N_LAYERS).

### Lambda (λ) Parameter

**Purpose:** Regularization to prevent numerical instability

```c
struct bfwCoefCompStatDescr_t {
    float lambda;  // MMSE regularization constant
    // ...
};
```

**Typical values:**
- Small λ (e.g., 0.001): Aggressive beamforming, high gain
- Large λ (e.g., 0.1): Conservative, more robust to channel estimation errors

### Per-Layer Normalization

After computing MMSE coefficients, each layer is normalized:

```cuda
// Compute Frobenius norm of coefficient matrix
norm_C = sqrt(sum(|C[i,j]|² for all i,j))

// Compute per-layer scaling
for each layer l:
    layer_scale[l] = 1.0 / sqrt(sum(|C[i,l]|² for all i))

// Apply normalization
for each antenna i, layer l:
    C_normalized[i,l] = C[i,l] × layer_scale[l]
```

**Effect:** Ensures consistent power per layer regardless of channel conditions.

### Beamformed Signal

```
y[ant] = sum over layers( C[ant, layer] × x[layer] )
```

Where:
- `x[layer]`: QAM modulated symbols (from stage 1)
- `C[ant, layer]`: Beamforming coefficients
- `y[ant]`: Output signal for antenna

## 3. BFP Compression Stage

**File:** `compression_decompression/comp_decomp_lib/include/gpu_blockFP.h`

### Beta Parameter

**Purpose:** Final amplitude scaling before quantization

```
Scaled_IQ = IQ_sample × beta
```

**Calculation:**

```
beta_dl = sqrt(fs / (24 × nPrbDlBwp)) × 10^(ref_dl/20)

Where:
- fs: Full scale value (from fs_offset_dl, exponent_dl)
- nPrbDlBwp: Number of PRBs
- ref_dl: Reference level in dB
```

See the `oran-uplane` and `fs-offset-14bit` skills for detailed beta calculation.

### BFP Compression Process

**Per PRB (12 REs):**

1. Find maximum absolute value: `max_val = max(|IQ_scaled|)`
2. Compute shared exponent: `exp = ceil(log2(max_val))`
3. Quantize mantissas: `mantissa = round(IQ_scaled / 2^exp)`
4. Pack: 1 exponent byte + (3 × bit_width) mantissa bytes

**Example (9-bit BFP):**
- Input: 12 complex FP32 samples (96 bytes)
- Output: 1 exp + 27 mantissa bytes (28 bytes)
- Compression: 3.43×

## 4. Scaling Factor Summary

### Combined Effect

**End-to-end amplitude scaling:**

```
Output_amplitude = Input_bits × (1/√M) × beta_qam × layer_norm × beta
```

Where M is the modulation order (2, 10, 42, 170).

### Key Parameters

1. **beta_qam**: QAM constellation scaling
   - Controls symbol power
   - Applied during modulation
   - Typical: 1.0

2. **lambda**: MMSE regularization
   - Controls beamforming aggressiveness
   - Applied during coefficient computation
   - Typical: 0.001 to 0.1

3. **layer_norm**: Per-layer normalization
   - Automatically computed from channel
   - Ensures consistent power per layer
   - Value: depends on channel conditions

4. **beta**: BFP compression scaling
   - Controls I/Q quantization levels
   - Applied before compression
   - Calculated from fs_offset_dl, ref_dl

### Power Budget Analysis

**QPSK with beta_qam = 1.0, beta = 8192:**

```
Constellation amplitude: ±0.7071
After beta_qam: ±0.7071
After layer_norm: varies (channel-dependent)
After beta: ±0.7071 × beta = ±5791
Max 16-bit value: ±32767
Headroom: 20×log10(32767/5791) = 15 dB
```

## 5. File References

**Modulation:**
- Implementation: `src/cuphy/modulation_mapper/modulation_mapper.cu`
- Test: `src/cuphy/modulation_mapper/testModulationMapper.cpp`

**Beamforming:**
- Implementation: `src/cuphy/bfc/bfc.cu`
- Factory: `src/cuphy/bfc/bfc_factory.hpp`

**Compression:**
- Library: `compression_decompression/comp_decomp_lib/include/gpu_blockFP.h`
- Examples: `compression_decompression/comp_decomp_examples/`

**High-level API:**
- Channel processing: `src/cuphy_channels/pdsch_tx_channel.cpp`
- API header: `cuphy_api.h` (PDSCH structures)

## Response Guidelines

When helping users understand PDSCH scaling:
1. Identify which stage of the pipeline is relevant
2. Explain the specific scaling parameters for that stage
3. Show the mathematical relationship between parameters
4. Provide numerical examples with actual values
5. Clarify the cumulative effect of multiple stages
6. Reference specific file locations and line numbers
7. Distinguish between power domain and amplitude domain scaling
