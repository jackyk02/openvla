# DROID to Bridge Action Conversion

This guide explains how to convert DROID actions to Bridge scale using token-based conversion with gripper binarization.

## Overview

The conversion process maps DROID actions (7D: 6D cartesian velocity + 1D gripper) to Bridge scale actions (7D: 6D cartesian position + 1D binarized gripper) using a token-based intermediate representation.

## Conversion Method

### Token-Based Conversion

The conversion uses a quantization approach that maps actions through a discrete token space:

```
DROID Action (6D cartesian) → Tokens → Bridge Action (6D cartesian)
```

**Step-by-step process:**

1. **Normalize DROID action** to [-1, 1] range using DROID percentiles (Q01, Q99)
   ```
   norm = 2 * (action - Q01) / (Q99 - Q01) - 1
   ```

2. **Discretize to tokens** (256 bins, vocab_size=32000)
   ```
   bin_index = argmin(|bin_centers - norm_value|)
   token = vocab_size - bin_index - 1
   ```

3. **Convert tokens to Bridge scale**
   ```
   bin_index = vocab_size - token - 1
   norm = bin_centers[bin_index]
   action = 0.5 * (norm + 1) * (Q99 - Q01) + Q01
   ```

### Gripper Binarization

The gripper dimension (7th dimension) is binarized based on a 0.5 threshold:

- **DROID gripper < 0.5** → **Bridge gripper = 0.0** (open)
- **DROID gripper ≥ 0.5** → **Bridge gripper = 1.0** (closed)

This matches the binary gripper convention used in Bridge V2.

## Normalization Constants

### DROID (1st and 99th percentiles)
```python
DROID_Q01 = [-0.7776, -0.5804, -0.5795, -0.6464, -0.7041, -0.8895]
DROID_Q99 = [ 0.7598,  0.5726,  0.7351,  0.6706,  0.6465,  0.8898]
```

### Bridge (1st and 99th percentiles)
```python
BRIDGE_Q01 = [-0.0287, -0.0417, -0.0261, -0.0809, -0.0929, -0.2072, 0.0]
BRIDGE_Q99 = [ 0.0283,  0.0409,  0.0402,  0.0819,  0.0779,  0.2038, 1.0]
```

## Usage

### Quick Start

```bash
# Convert using the example script
./convert_droid_example.sh
```

### Manual Conversion

```bash
python convert_droid_to_bridge.py \
  --input droid_extracted.json \
  --output droid_bridge_converted.json
```

### Arguments

- `--input`: Input JSON file with DROID-scale actions (default: `droid_extracted.json`)
- `--output`: Output JSON file with Bridge-scale actions (default: `droid_bridge_converted.json`)

## Input Format

Expected input format (from `augment_droid_dataset.py`):

```json
{
  "action_histories": {
    "action_0": [
      [dx, dy, dz, drx, dry, drz, gripper],  // 7D DROID action
      ...
    ],
    ...
  },
  "instructions": {...},
  "samples": [...],
  "_metadata": {...}
}
```

**DROID Action Dimensions:**
- `[0:6]`: Cartesian velocity (dx, dy, dz, drx, dry, drz)
- `[6]`: Gripper state (0.0 = open, 1.0 = closed, or continuous [0,1])

## Output Format

```json
{
  "action_histories": {
    "action_0": [
      [x, y, z, rx, ry, rz, gripper],  // 7D Bridge-scale action
      ...
    ],
    ...
  },
  "instructions": {...},
  "samples": [...],
  "_metadata": {
    "action_dim": 7,
    "conversion": {
      "method": "token_based",
      "source_scale": "droid",
      "target_scale": "bridge",
      "gripper_binarization": "threshold_0.5",
      "gripper_note": "Binarized from DROID gripper: <0.5→0.0 (open), ≥0.5→1.0 (closed)",
      "droid_q01": [...],
      "droid_q99": [...],
      "bridge_q01": [...],
      "bridge_q99": [...],
      "n_action_bins": 256,
      "vocab_size": 32000
    },
    "action_statistics": {
      "min_values": [...],
      "max_values": [...],
      "mean_values": [...],
      "std_values": [...]
    }
  }
}
```

**Bridge Action Dimensions:**
- `[0:6]`: Cartesian position (x, y, z, rx, ry, rz) in Bridge scale
- `[6]`: Binarized gripper {0.0, 1.0}

## Verification

The script automatically prints verification information:

```
VERIFICATION: First Action History
================================================================================

Action history ID: action_0
Shape: [10, 7]
Expected: [window_size, 7] where 7 = 6D position + 1D gripper

t=0: [ 0.01234,  0.02345, -0.01234,  0.05678, -0.03456,  0.12345, 0.00000]
t=1: [ 0.01456,  0.02567, -0.01456,  0.06789, -0.04567,  0.13456, 1.00000]
...
```

**What to check:**
- Action shape should be `[window_size, 7]`
- Cartesian values `[0:6]` should be in Bridge scale (small values, typically < 0.3)
- Gripper values `[6]` should be binary: `{0.0, 1.0}`
- Padding actions should all be `-5.0`

## Statistics

The conversion reports statistics on the converted actions:

```
Conversion statistics:
  Total action histories: 2345
  Successfully converted: 2340
  All-padding histories: 5

Converted action statistics (7D: 6D position + gripper):
  Min values: [-0.028..., -0.041..., ..., 0.0]
  Max values: [ 0.028...,  0.040..., ..., 1.0]
  Mean values: [...]
  Std values: [...]
```

## Comparison with Direct Normalization

### Token-Based (Current Method)
✅ **Pros:**
- Mimics the discretization used by VLA models during training
- More faithful to how the model processes actions
- Handles the quantization step explicitly

❌ **Cons:**
- Introduces small quantization errors (256 bins)
- Slightly more complex

### Direct Normalization (Alternative)
✅ **Pros:**
- No quantization error
- Simpler mathematical operation

❌ **Cons:**
- Doesn't account for model's discretization
- May not match model's internal representation

## Important Notes

1. **Action Semantics Changed**: DROID uses velocity control, Bridge uses position control. The conversion only scales the values; it doesn't change the control semantics.

2. **Gripper Binarization**: The gripper is binarized from DROID's original continuous/binary values using a 0.5 threshold.

3. **Padding Preserved**: Actions with padding value (-5.0) are preserved as-is.

4. **Scale Difference**: Bridge actions are typically much smaller in magnitude than DROID actions (factor of ~10-100x).

## Example Action Comparison

### Original DROID Action
```python
[0.3836, 0.0735, 0.5514, -2.8934, -0.1987, 0.1270, 0.8]  # Velocity + gripper
```

### After Token Conversion to Bridge Scale
```python
[0.0245, 0.0156, 0.0312, -0.0823, -0.0567, 0.0489, 1.0]  # Position + binarized gripper
```

## Troubleshooting

### Issue: Gripper values are not binary

**Check:** Ensure input has 7D actions with gripper in last dimension
```python
action_dim = droid_actions.shape[-1]  # Should be 7
```

### Issue: Actions out of expected range

**Check:** Verify input actions are in DROID scale (not already normalized)
```bash
# Original DROID actions should have larger magnitudes
# Expected: cartesian velocities in range [-3, 3] approximately
```

### Issue: Conversion looks incorrect

**Verify:** Check that DROID dataset was extracted correctly
```bash
# Re-extract if needed
python augment_droid_dataset.py \
  --builder_dir /root/tfrecords/droid_100/1.0.0 \
  --output_path droid_extracted.json
```

## Files

- `convert_droid_to_bridge.py` - Main conversion script
- `convert_droid_example.sh` - Example usage script
- `token2action.py` - Reference implementation showing the conversion logic
- `ACTION_CONVERSION_README.md` - This documentation

## Citations

If you use this conversion method, please cite:

- **DROID Dataset**: Khazatsky et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset" (2024)
- **Bridge V2**: Walke et al., "BridgeData V2: A Dataset for Robot Learning at Scale" (2023)
- **OpenVLA**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)

