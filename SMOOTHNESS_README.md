# Movement Smoothness Analysis - Dimensionless Jerk

## Overview

A new smoothness metric based on **dimensionless jerk** has been added to the movement analysis pipeline. This metric quantifies how smooth or jerky a movement trajectory is, providing insights into motor control quality.

## What is Dimensionless Jerk?

Dimensionless jerk is a normalized measure of movement smoothness that calculates the third derivative of position (jerk) over a movement trajectory. The metric is:

- **Scale-independent**: Normalized by movement duration and path length
- **Lower values = smoother movement**: More controlled, less jerky movements
- **Higher values = less smooth movement**: More variable, jerky movements

## Mathematical Definition

The dimensionless jerk is calculated as:

```
DJ = sqrt((T^5 / PL^2) * ∫(jerk²) dt)
```

Where:
- `T` = movement duration (seconds)
- `PL` = path length (total distance traveled)
- `jerk` = third derivative of position (rate of change of acceleration)

## Implementation Details

### Function Location
- **File**: `utils.py`
- **Function**: `dimensionless_jerk(x, y, t)`

### Parameters
- `x`: Array of x-coordinates
- `y`: Array of y-coordinates  
- `t`: Array of time points

### Integration in Analysis Pipeline

The smoothness metric is calculated in the `compute_trial()` function:

1. **Movement Segment**: From movement onset (RT) to first target entry
2. **Minimum Data**: Requires at least 4 data points for calculation
3. **Output Column**: `movement_smoothness` in the CSV results

### CSV Output

The new metric appears as the `movement_smoothness` column in:
- `resume_resultats.csv`
- `updated_resume_resultats.csv`

## Interpretation Guidelines

### Typical Values
- **Very smooth movements**: < 1.0
- **Moderately smooth**: 1.0 - 5.0
- **Jerky movements**: > 5.0
- **Very jerky movements**: > 10.0

### Clinical/Research Applications
- **Motor control assessment**: Lower values indicate better motor control
- **Pathology detection**: Higher values may indicate movement disorders
- **Training effectiveness**: Improvements should show decreasing values over time
- **Fatigue assessment**: Increasing values may indicate motor fatigue

## Error Handling

The function returns `None` in the following cases:
- Insufficient data points (< 4)
- Invalid time series (non-increasing timestamps)
- Zero movement duration or path length
- Calculation errors (division by zero, etc.)

## Usage Example

```python
from utils import dimensionless_jerk
import numpy as np

# Example trajectory
x = np.array([0, 10, 20, 30, 40])
y = np.array([0, 5, 10, 15, 20])
t = np.array([0, 0.1, 0.2, 0.3, 0.4])

smoothness = dimensionless_jerk(x, y, t)
print(f"Movement smoothness: {smoothness}")
```

## Validation

Use the `validate_smoothness.py` script to test the implementation:

```bash
python validate_smoothness.py
```

This script tests various movement patterns and validates that the function produces reasonable results.

## References

This implementation is based on established movement analysis literature:
- Hogan, N., & Sternad, D. (2009). Sensitivity of smoothness measures to movement duration, amplitude, and arrests. Journal of Motor Behavior, 41(6), 529-534.
- Flash, T., & Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. Journal of Neuroscience, 5(7), 1688-1703.
