---
name: prach-processing
description: Expert on PRACH uplink processing chain from O-RAN U-Plane reception through GPU detection to FAPI RACH indication. Use when discussing PRACH signal processing, random access, timing advance, or PHY-MAC interface.
---

# PRACH Uplink Processing Chain Expert

You are an expert on PRACH (Physical Random Access Channel) uplink processing in NVIDIA Aerial cuBB, covering the complete flow from O-RAN fronthaul reception to FAPI RACH indication delivery.

## Core Expertise

### 1. PRACH Processing Pipeline Overview

Complete processing chain:
```
O-RU U-Plane → Fronthaul Driver → Order Entity → PhyPrachAggr →
cuPHY PRACH RX (GPU) → Callback → FAPI Builder → MAC Transport
```

**Processing Stages:**
1. **O-RAN Packet Reception**: DPDK-based U-Plane packet capture
2. **Decompression & Ordering**: IQ sample extraction and GPU memory preparation
3. **GPU Detection**: FFT, correlation, preamble detection, timing/power estimation
4. **FAPI Formatting**: Standards-compliant RACH indication message construction
5. **MAC Delivery**: Zero-copy transport to MAC layer

**Typical Latency Budget:**
- O-RU Reception → Order Entity: ~0.5 ms
- GPU Processing: ~0.7 ms
- FAPI Construction & Transport: ~0.1 ms
- **Total End-to-End**: ~1.2 ms

### 2. Key Software Components

**Component Hierarchy:**

| Component | File Path | Responsibility |
|-----------|-----------|----------------|
| **Fronthaul Driver** | `aerial-fh-driver/lib/aerial_fh_driver.cpp` | O-RAN packet reception |
| **Stream RX** | `aerial-fh-driver/lib/stream_rx.cpp` | DPDK queue management |
| **Order Entity** | `cuphydriver/src/uplink/order_entity.cpp` | IQ decompression/ordering |
| **Slot Map UL** | `cuphydriver/include/slot_map_ul.hpp` | Task scheduling |
| **PhyPrachAggr** | `cuphydriver/src/uplink/phyprach_aggr.cpp` | PRACH orchestration |
| **cuPHY PRACH RX** | `cuPHY/src/cuphy_channels/prach_rx.cpp` | GPU detection engine |
| **FAPI Builder** | `scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp:3286` | Message formatting |

### 3. PhyPrachAggr Class (Main Orchestrator)

**File**: `cuphydriver/include/phyprach_aggr.hpp`

**Key Data Members:**
```cpp
// cuPHY handle
cuphyPrachRxHndl_t handle;

// Input tensors (IQ data from Order Entity)
cuphyTensorPrm_t tDataRx;
cuphy::tensor_desc prach_data_rx_desc[PRACH_MAX_OCCASIONS_AGGR];

// Output tensors (GPU device memory)
cuphy::tensor_device gpu_num_detectedPrmb;        // Preambles detected
cuphy::tensor_device gpu_prmbIndex_estimates;     // Preamble indices [0-63]
cuphy::tensor_device gpu_prmbDelay_estimates;     // Timing advance (μs)
cuphy::tensor_device gpu_prmbPower_estimates;     // Power (linear)

// Output tensors (CPU pinned memory for DMA)
cuphy::tensor_pinned cpu_num_detectedPrmb;
cuphy::tensor_pinned cpu_prmbIndex_estimates;
cuphy::tensor_pinned cpu_prmbDelay_estimates;
cuphy::tensor_pinned cpu_prmbPower_estimates;
cuphy::tensor_pinned ant_rssi;                    // Per-antenna RSSI
cuphy::tensor_pinned rssi;                        // Average RSSI
cuphy::tensor_pinned interference;                // Interference power

// Configuration
cuphyPrachStatPrms_t prach_params_static;
cuphyPrachOccaDynPrms_t* prach_dyn_occa_params;
```

**Processing Flow:**
```cpp
void PhyPrachAggr::run(cudaStream_t stream) {
    // 1. Setup input descriptors
    // 2. Call cuphySetupPrachRx() with config
    // 3. Call cuphyRunPrachRx() for GPU processing
    // 4. DMA results from GPU to CPU (async)
    // 5. Synchronize stream
}
```

### 4. PRACH Configuration Parameters

**Static Configuration** (`cuphyPrachStatPrms_t`):
```cpp
struct cuphyPrachCellStatPrms_t {
    uint32_t N_ant;                    // Number of antennas (e.g., 4)
    uint32_t mu;                       // Subcarrier spacing (0=15kHz, 1=30kHz, 2=60kHz)
    uint32_t configurationIndex;       // PRACH config (0-255, per TS 38.211)
    cuphyFreqRange_t freqRange;        // FR1 or FR2
    cuphyDuplexMode_t duplexMode;      // TDD or FDD
    cuphyPrachRestrictedSet_t restrictedSetCfg; // Unrestricted/Type A/Type B
};

struct cuphyPrachOccaStatPrms_t {
    uint32_t rootSequenceIndex;        // Root sequence (0-837)
    uint32_t zeroCorrelationZoneCfg;   // ZCZ config (0-15)
};
```

**Dynamic Configuration** (`cuphyPrachDynPrms_t`):
```cpp
struct cuphyPrachOccaDynPrms_t {
    float forceThreshold;              // Detection threshold (dB)
    uint32_t freqIndex;                // Frequency index (0-7)
    uint32_t symbolIndex;              // Starting symbol (0-13)
};
```

**Key Parameters:**
- **Root Sequence Index (0-837)**: Determines Zadoff-Chu sequence base
- **Configuration Index**: Defines PRACH format, duration, subcarrier spacing
  - Example: Index 0 = Format 0, 839 samples, 15 kHz SCS
  - Example: Index 67 = Format A1, 139 samples, 30 kHz SCS
- **ZCZ Config**: Sets cyclic shift spacing (N_cs values 0-30)
- **Detection Threshold**: Peak detection threshold in dB

### 5. GPU Detection Algorithm (cuPHY PRACH RX)

**Processing Steps:**
```
1. FFT: Time domain → Frequency domain
   - Per-antenna transformation
   - CUFFT library
   - Output: Frequency-domain IQ samples

2. Reference Sequence Generation
   - Zadoff-Chu sequences: x(n) = exp(-j × π × u × n × (n+1) / L_RA)
   - u = root sequence index
   - L_RA = 139 or 839 samples
   - 64 possible preambles (cyclic shifts)

3. Cross-Correlation
   - Correlate RX signal with each preamble reference
   - Per-antenna processing
   - Output: Correlation peaks per preamble

4. Multi-Antenna Combining
   - Coherent or non-coherent combining
   - Power summation across antennas

5. Peak Detection
   - Compare against threshold
   - Local maxima finding
   - Output: Detected preamble indices

6. Delay Estimation
   - Peak position → timing offset
   - Convert to microseconds
   - Resolution: ~1 sample period

7. Power Measurement
   - Correlation peak magnitude
   - Linear power value
   - Per-preamble measurement

8. RSSI/Interference Calculation
   - Average received signal strength
   - Noise floor estimation
   - Per-antenna and aggregate metrics
```

**Output Structure:**
```cpp
struct cuphyPrachStatusOut_t {
    uint32_t* num_detectedPrmb;      // [nOccasions] - 0 to 64 per occasion
    uint32_t* prmbIndex_estimates;   // [nOccasions × MAX_PREAMBLES] - Index 0-63
    float* prmbDelay_estimates;      // [nOccasions × MAX_PREAMBLES] - Delay in μs (0-1300)
    float* prmbPower_estimates;      // [nOccasions × MAX_PREAMBLES] - Linear power
    float* ant_rssi;                 // [nOccasions × nAntennas] - RSSI in dB
    float* rssi;                     // [nOccasions] - Average RSSI in dB
    float* interference;             // [nOccasions] - Interference power in dBm
};
```

### 6. FAPI Message Construction

**File**: `scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`

**Callback Registration** (Line 4636):
```cpp
cb.ul_cb.prach_cb_fn = [this] (
    slot_command_api::slot_indication& slot,
    const prach_params& params,
    const uint32_t* num_detectedPrmb,
    const void* prmbIndex_estimates,
    const void* prmbDelay_estimates,
    const void* prmbPower_estimates,
    const void* ant_rssi,
    const void* rssi,
    const void* interference)
{
    send_rach_indication(...);
};
```

**FAPI Message Structure** (`scf_5g_fapi.h`):
```cpp
// Message ID: 0x89
typedef struct {
    scf_fapi_body_header_t msg_hdr;
    uint16_t sfn;                    // System Frame Number (0-1023)
    uint16_t slot;                   // Slot (0-159)
    uint8_t num_pdus;                // Number of PRACH occasions
    scf_fapi_prach_ind_pdu_t pdu_info[0];
} scf_fapi_rach_ind_t;

typedef struct {
    uint16_t phys_cell_id;           // Physical Cell ID (0-1007)
    uint8_t symbol_index;            // Starting symbol
    uint8_t slot_index;              // Slot index
    uint8_t freq_index;              // Frequency index
    uint8_t avg_rssi;                // RSSI [0-254] = [-63 to +30] dB
    uint8_t avg_snr;                 // SNR (0xFF if not computed)
    uint8_t num_preamble;            // Detected preambles (0-64)
    scf_fapi_prach_preamble_info_t preamble_info[0];
} scf_fapi_prach_ind_pdu_t;

typedef struct {
    uint8_t preamble_index;          // Preamble ID (0-63)
    uint16_t timing_advance;         // TA [0-3846] in units of 64×16×Tc
    uint32_t preamble_power;         // Power [0-170000] in 0.001 dB steps
} scf_fapi_prach_preamble_info_t;
```

### 7. Critical Data Conversions

**RSSI Conversion** (Line 3378):
```cpp
// Input: RSSI in dB (float, range -63 to +30 dB)
// Output: FAPI format (uint8_t, range 0-254)
// Formula: RSSI_FAPI = (RSSI_dB × 2) + 128
pdu.avg_rssi = rssi[i] * 2 + 128 + 0.5;

// Example:
//   -63 dB → 2 (min valid)
//     0 dB → 128 (midpoint)
//   +30 dB → 254 (max valid)
// Resolution: 0.5 dB per step
```

**Timing Advance Conversion** (Lines 3412-3419):
```cpp
// Input: Delay in microseconds (float)
// Output: FAPI TA units (uint16_t, 0-3846)
// Units: 64 × 16 × Tc, where Tc = 1/(480000×4096) seconds

constexpr float STEP_CONST = 0.509803921568627; // From nv::STEP_CONST
float step = STEP_CONST / (1 << params.mu);      // Adjust for subcarrier spacing

// Convert μs to FAPI units, accounting for TA offset
int tmp = (delay_time_est[prmb_idx] - prach_ta_offset_usec_/1000000.0) *
          (1 << phy_module().get_mu_highest()) * 192 * 10000;

preamble.timing_advance = std::clamp(tmp, 0, 3846);

// Range: 0-3846 represents 0 to ~1300 μs
// Max cell radius: ~100 km (round-trip delay)
```

**Preamble Power Conversion** (Line 3422):
```cpp
// Input: Linear power (correlation peak magnitude squared)
// Output: FAPI format [0-170000] representing [-140 to +30] dBm in 0.001 dB steps

constexpr float RACH_BB2RF_powerOffset = -48.68; // dB calibration offset

// Convert to dBm with offset
float power_dbm = 10 * std::log10(peak_dest[prmb_idx]) + 140 + RACH_BB2RF_powerOffset;

// Convert to FAPI units (0.001 dB resolution)
preamble.preamble_power = power_dbm * 1000 + 0.5;

// Clamp to valid range
preamble.preamble_power = std::clamp(preamble.preamble_power, 0u, 170000u);

// Example:
//   -140 dBm → 0
//   -50 dBm  → 90000
//   0 dBm    → 140000
//   +30 dBm  → 170000
```

### 8. PRACH Formats and Configuration

**Common PRACH Formats** (3GPP TS 38.211):

| Format | Duration | Samples | Subcarrier Spacing | Use Case |
|--------|----------|---------|-------------------|----------|
| 0 | 1 ms | 839 | 15 kHz | Normal coverage |
| 1 | 0.67 ms | 839 | 15 kHz | Normal coverage |
| A1 | 35.3 μs | 139 | 30 kHz | Short format, low latency |
| A2 | 70.6 μs | 139 | 30 kHz | Short format |
| B1 | 100 μs | 139 | 60 kHz | Short format, FR1/FR2 |
| C0 | 1 ms | 1151 | 15 kHz | Long format, extended coverage |
| C2 | 2 ms | 2 × 1151 | 15 kHz | Extended coverage |

**Configuration Index Examples:**
- Index 0: Format 0, 1 occasion per slot
- Index 16: Format 1, 2 occasions per slot
- Index 67: Format A1, 2 occasions per slot (30 kHz SCS)

**Root Sequence and ZCZ:**
- **Root Sequence (0-837)**: Defines base Zadoff-Chu sequence
- **Zero Correlation Zone**: Controls cyclic shift spacing
  - ZCZ Config 0: N_cs = 0 (maximum preambles, minimum separation)
  - ZCZ Config 15: N_cs = 30 (fewer preambles, higher separation for high-speed UEs)

### 9. Performance Metrics

**GPU Processing Throughput:**
```
Per Occasion (4 antennas, Format 0):
  - FFT:          ~50 μs
  - Correlation:  ~100 μs
  - Detection:    ~20 μs
  - Total:        ~170 μs

4 Occasions (typical slot):
  - GPU kernels:  ~680 μs
  - DMA D2H:      ~50 μs
  - Total:        ~730 μs
```

**Memory Footprint per Cell:**
```
Input Buffers:        ~52 KB (GPU device)
Workspace:            ~2 MB (GPU device)
Output Buffers:       ~4 KB (GPU + CPU pinned)
Reference Sequences:  ~200 KB (GPU, cached)
Total:                ~2.3 MB per cell
```

**Latency Budget:**
```
O-RAN Packet RX:      ~100 μs
Decompression:        ~200 μs
GPU Processing:       ~680 μs
DMA Transfer:         ~50 μs
FAPI Construction:    ~50 μs
Transport Send:       ~20 μs
--------------------------------
Total End-to-End:     ~1.1 ms
```

### 10. Debugging and Validation

**Key Log Points:**
```cpp
// Enable PRACH debug logging
export CUPHY_LOG_LEVEL=DEBUG
export PRACH_DEBUG=1
```

**Validation Checks:**
```cpp
void PhyPrachAggr::validate() {
    for (uint32_t i = 0; i < nOccasions; i++) {
        // Check detection count
        assert(cpu_num_detectedPrmb[i] <= 64);

        for (uint32_t j = 0; j < cpu_num_detectedPrmb[i]; j++) {
            uint32_t idx = i * MAX_PREAMBLES + j;

            // Validate preamble index (0-63)
            assert(cpu_prmbIndex_estimates[idx] < 64);

            // Validate timing advance (0-1300 μs)
            assert(cpu_prmbDelay_estimates[idx] >= 0.0f);
            assert(cpu_prmbDelay_estimates[idx] <= 1300.0f);

            // Validate power estimate
            float power_db = 10*log10(cpu_prmbPower_estimates[idx]);
            assert(power_db >= -140.0f && power_db <= 30.0f);
        }
    }
}
```

**Test Vector Support:**
- Location: `cuPHY/test/test_vectors/prach/`
- Format: HDF5 files with input IQ, expected detections, reference correlations

### 11. Common Issues and Troubleshooting

**Issue 1: No Preambles Detected**
- Check: Detection threshold (forceThreshold) too high
- Check: Root sequence index mismatch with UE configuration
- Check: IQ sample power levels (RSSI)
- Check: Compression artifacts (BFP bit-width too low)

**Issue 2: False Detections**
- Check: Detection threshold too low
- Check: Interference levels
- Check: ZCZ configuration (cyclic shift spacing)

**Issue 3: Incorrect Timing Advance**
- Check: `prach_ta_offset_usec_` calibration value
- Check: Subcarrier spacing (mu) parameter
- Check: Sample rate alignment with PRACH format

**Issue 4: Power Estimation Errors**
- Check: `RACH_BB2RF_powerOffset` calibration (-48.68 dB)
- Check: Antenna calibration factors
- Check: Decompression scaling (beta_ul)

### 12. Integration Points

**Order Entity Interface:**
```cpp
// Ordered PRACH PRBs ready for GPU processing
cuphy::tensor_device ordered_prbs_prach[MAX_CELLS];

// Synchronization flag
cuphy::buffer<uint8_t> order_kernel_exit_cond_gdr;
```

**Slot Map UL Interface:**
```cpp
class SlotMapUl {
    PhyPrachAggr* aggr_prach;          // PRACH processor
    ULInputBuffer* input_buffer;        // Input buffer pool
    nv::hrc::time_point start_t_ul_prach_cuda;
    nv::hrc::time_point end_t_ul_prach_cuda;
};
```

**Transport Layer:**
```cpp
// Message allocation and transmission
auto msg_desc = transport.tx_alloc(msg_size, cell_id);
transport.tx_send(msg_desc);
transport.notify(1);
```

## When to Use This Skill

Invoke this skill when users ask about:
- PRACH uplink processing flow or architecture
- Random access channel detection and processing
- Timing advance estimation and calculation
- Preamble detection algorithms
- PRACH configuration (root sequence, ZCZ, format)
- FAPI RACH indication message format
- GPU-accelerated PRACH processing
- PHY-MAC interface for random access
- O-RAN fronthaul PRACH reception
- cuPHY PRACH RX API usage
- PRACH performance optimization
- Debug and validation of PRACH processing

## Response Guidelines

- Reference specific file paths with line numbers when applicable
- Provide concrete examples with numerical values
- Explain data conversions between cuPHY outputs and FAPI formats
- Show configuration examples for different PRACH formats
- Include timing diagrams and latency breakdowns
- Reference 3GPP specifications (TS 38.211, 38.213) when relevant
- Clarify the complete processing pipeline from O-RU to MAC
- Highlight GPU optimization strategies and performance metrics
