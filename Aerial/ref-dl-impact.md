---
name: ref-dl-impact
description: Quick reference for ref_dl parameter impact on beta_dl. Use when users ask about ref_dl values, amplitude scaling relationships, or how to maintain constant effective bits with different ref_dl settings.
---

# ref_dl Parameter Impact Calculator

You are an expert on the `ref_dl` parameter and its impact on `beta_dl` scaling in O-RAN U-Plane I/Q level setting.

## Quick Answer

**Increasing `ref_dl` by X dB increases `beta_dl` by exactly X dB (in amplitude).**

```
beta_dl = sqrt(fs / (24 × nPrbDlBwp)) × 10^(ref_dl/20)
```

## Key Insight

```
10^(ref_dl/10) = Power domain scaling
sqrt(10^(ref_dl/10)) = 10^(ref_dl/20) = Amplitude domain scaling

In amplitude dB: 20 × log10(10^(ref_dl/20)) = ref_dl dB
```

**The square root is already accounted for in the dB conversion formula.**

## Numerical Examples

### Example Configuration
- dl_bit_width = 9
- fs_offset_dl = 7.3
- exponent_dl = 4
- nPrbDlBwp = 273

### Impact Table

| ref_dl | Power Factor | beta_dl | Ratio to 0dB | Increase (dB) |
|--------|--------------|---------|--------------|---------------|
| 0 dB | 1.0000 | 8,256 | 1.00× | 0 dB |
| 1 dB | 1.2589 | 9,263 | 1.12× | 1.00 dB |
| 2 dB | 1.5849 | 10,393 | 1.26× | 2.00 dB |
| 3 dB | 1.9953 | 11,661 | 1.41× | 3.00 dB |
| 6 dB | 3.9811 | 16,472 | 2.00× | 6.00 dB |
| 10 dB | 10.0000 | 26,106 | 3.16× | 10.00 dB |

**Verification:**
```
10^(3/20) = 1.4125 ✓
20 × log10(11,661 / 8,256) = 3.00 dB ✓
```

## Common Misconception

❌ **WRONG:** "ref_dl = 3 dB increases beta by 1.5 dB because sqrt halves the dB"

```
Incorrect reasoning:
10^(3/10) = power (2.0×)
sqrt(2.0) = 1.41× in amplitude
20 × log10(1.41) = 1.5 dB ← WRONG!
```

✅ **CORRECT:** "ref_dl = 3 dB increases beta by 3.0 dB in amplitude"

```
Correct reasoning:
sqrt(10^(ref_dl/10)) = 10^(ref_dl/20)
For ref_dl = 3 dB:
10^(3/20) = 1.4125× in amplitude
20 × log10(1.4125) = 3.0 dB ← CORRECT!
```

**Why the confusion?** The conversion from power dB (factor of 10) to amplitude dB (factor of 20) already accounts for the square root.

## Maintaining Constant Effective Bits

To keep the same effective bits when changing ref_dl, adjust fs_offset_dl:

### Compensation Table

| ref_dl | fs_offset_dl | beta_dl | Effective Bits |
|--------|--------------|---------|----------------|
| 0 dB | 7.3 | 8,192 | 14.00 |
| 1 dB | 7.5 | 8,192 | 14.00 |
| 2 dB | 7.8 | 8,192 | 14.00 |
| 3 dB | 8.3 | 8,192 | 14.00 |
| 6 dB | 9.3 | 8,192 | 14.00 |

### Compensation Rule

```
Δfs_offset_dl ≈ Δref_dl / 3

Example:
ref_dl increases by 3 dB
→ fs_offset_dl increases by ~1.0
→ beta_dl remains constant
```

## Practical Use Cases

### When to Increase ref_dl

**Scenario:** Lower signal levels in the system

- Need more amplification
- System reference power is low
- Want to boost beta without changing fs_offset_dl

**Action:** Increase ref_dl (e.g., from 0 to 3 dB)

**Effect:** beta increases proportionally (3 dB increase)

### When to Decrease ref_dl

**Scenario:** Higher signal levels in the system

- Want to avoid over-driving
- System reference power is high
- Need to reduce beta without changing fs_offset_dl

**Action:** Decrease ref_dl (e.g., from 3 to 0 dB)

**Effect:** beta decreases proportionally (3 dB decrease)

### Coordinated Adjustment

**Objective:** Maintain constant beta (and effective bits)

```
Strategy:
1. Increase ref_dl by X dB (for system-level reasons)
2. Increase fs_offset_dl by X/3 (to compensate)
3. Result: beta remains constant
```

**Example:**
```
Initial: ref_dl = 0 dB, fs_offset_dl = 7.3 → beta = 8,192
Change ref_dl to 3 dB for system alignment
Adjust fs_offset_dl to 8.3 (increase by 1.0)
Final: ref_dl = 3 dB, fs_offset_dl = 8.3 → beta = 8,192
```

## Mathematical Verification

### Starting from Beta Formula

```
numerator = fs × 10^(ref_dl/10)
beta_dl = sqrt(numerator / (24 × nPrbDlBwp))
        = sqrt(fs / (24 × nPrbDlBwp)) × sqrt(10^(ref_dl/10))
        = sqrt(fs / (24 × nPrbDlBwp)) × 10^(ref_dl/20)
```

### Ratio Between Two ref_dl Values

```
beta_dl(ref2) / beta_dl(ref1) = 10^((ref2 - ref1)/20)

Example: ref1 = 0 dB, ref2 = 3 dB
Ratio = 10^(3/20) = 1.4125
In dB: 20 × log10(1.4125) = 3.0 dB
```

### Converting Power to Amplitude dB

```
Power scaling: P2/P1 = 10^(ΔdB_power/10)
Amplitude scaling: A2/A1 = sqrt(P2/P1) = sqrt(10^(ΔdB_power/10))
                                        = 10^(ΔdB_power/20)

In dB: ΔdB_amplitude = 20 × log10(10^(ΔdB_power/20))
                      = ΔdB_power

Conclusion: Same dB value, different interpretations!
- ΔdB in power domain uses factor of 10
- ΔdB in amplitude domain uses factor of 20
- But the numerical dB value is identical
```

## Quick Calculation

### To find beta with different ref_dl:

```
Given: beta_dl(ref1), want beta_dl(ref2)

beta_dl(ref2) = beta_dl(ref1) × 10^((ref2 - ref1)/20)
```

### To find ref_dl for target beta:

```
Given: beta_current, beta_target, ref_current

ref_target = ref_current + 20 × log10(beta_target / beta_current)
```

## File References

**Beta calculation:**
- `ru-emulator/ru_emulator/config_parser.cpp` - Beta computation with ref_dl

**Configuration:**
- `ru-emulator/config/*.yaml` - Set ref_dl parameter

## Response Guidelines

When helping users with ref_dl:
1. Emphasize the 1:1 relationship (X dB ref_dl → X dB beta amplitude)
2. Clarify the power vs amplitude dB conversion
3. Show numerical verification with examples
4. Explain compensation strategy with fs_offset_dl if needed
5. Provide practical use cases for adjusting ref_dl
