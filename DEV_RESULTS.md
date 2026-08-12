# Yin Experiment Results

> **Noise level:** 0.35 for all experiments and all random seeds.

## Faces

|              Seed |  Upright → Upright | Inverted → Inverted | Upright → Inverted |  Inverted → Upright |
| ----------------: | -----------------: | ------------------: | -----------------: | ------------------: |
|                 1 |             95.83% |              91.67% |             87.50% |              95.83% |
|                 2 |            100.00% |              87.50% |             87.50% |              91.67% |
|                 3 |             95.83% |              91.67% |             87.50% |              91.67% |
|                 4 |             95.83% |              83.33% |             87.50% |              79.17% |
|                 5 |             95.83% |              87.50% |             95.83% |              75.00% |
| **Mean (95% CI)** | **96.66% (±2.32)** |  **88.33% (±4.33)** | **89.17% (±4.63)** | **86.67% (±11.22)** |

## Objects

|              Seed |  Upright → Upright | Inverted → Inverted | Upright → Inverted |  Inverted → Upright |
| ----------------: | -----------------: | ------------------: | -----------------: | ------------------: |
|                 1 |             91.67% |              91.67% |             83.33% |              95.83% |
|                 2 |             83.33% |              91.67% |             87.50% |              87.50% |
|                 3 |             91.67% |              83.33% |             83.33% |              79.17% |
|                 4 |             91.67% |              95.83% |             75.00% |              83.33% |
|                 5 |             87.50% |              91.67% |             83.33% |             100.00% |
| **Mean (95% CI)** | **89.17% (±4.63)** |  **90.83% (±5.67)** | **82.50% (±5.67)** | **89.17% (±10.73)** |

## Houses

|              Seed |  Upright → Upright | Inverted → Inverted |  Upright → Inverted | Inverted → Upright |
| ----------------: | -----------------: | ------------------: | ------------------: | -----------------: |
|                 1 |             93.33% |             100.00% |              80.00% |             86.67% |
|                 2 |             93.33% |             100.00% |             100.00% |            100.00% |
|                 3 |            100.00% |              93.33% |             100.00% |             86.67% |
|                 4 |             93.33% |              93.33% |              86.67% |             86.67% |
|                 5 |            100.00% |             100.00% |              93.33% |             80.00% |
| **Mean (95% CI)** | **96.00% (±4.54)** |  **97.33% (±4.54)** | **92.00% (±10.79)** | **88.00% (±9.07)** |

*Confidence intervals are 95% t-based CIs across the five random seeds.*

---

# Thatcher Effect Results

> **Noise level:** 0.35 for all experiments.

The Thatcher experiment compares familiarity for normal and Thatcherized faces under upright and inverted presentation. The Thatcher effect is quantified as the difference in familiarity between normal and Thatcherized stimuli, with a larger positive value indicating greater sensitivity to Thatcherization.

## Non-Log-Polarized Representation

| Condition    |   Normal | Thatcher | Thatcher Effect |
| ------------ | -------: | -------: | --------------: |
| **Upright**  | -10.6873 | -12.0622 |      **1.3749** |
| **Inverted** | -25.4104 | -25.5089 |      **0.0985** |

The upright Thatcher effect was substantially larger than the inverted Thatcher effect:

* **Upright Thatcher effect:** 1.3749
* **Inverted Thatcher effect:** 0.0985
* **Difference:** 1.2764

A statistical test of the results yielded:

```text
t = 7.66396
p = 6.00988 × 10⁻¹²
df = 116
```

**Result: significant Thatcher effect.**

The strong reduction in the Thatcher effect for inverted faces is consistent with the characteristic orientation dependence of the Thatcher illusion.

## Log-Polarized Representation

| Condition    |    Normal |  Thatcher | Thatcher Effect |
| ------------ | --------: | --------: | --------------: |
| **Upright**  | -462.3432 | -505.6108 |     **43.2677** |
| **Inverted** | -643.0729 | -675.0939 |     **32.0210** |

The upright Thatcher effect was larger than the inverted Thatcher effect:

* **Upright Thatcher effect:** 43.2677
* **Inverted Thatcher effect:** 32.0210
* **Difference:** 11.2466

A statistical test of the results yielded:

```text
t = 3.38569
p = 0.000969659
df = 116
```

**Result: significant Thatcher effect.**

Thus, both representations exhibit a statistically significant Thatcher effect, although the magnitude and degree of orientation dependence differ substantially between the non-log-polarized and log-polarized representations.

---

# Kanwisher Results

> **Noise level:** 0.35 for all experiments.

The Kanwisher-style inversion experiment measured classification accuracy for upright and inverted presentations. The inversion effect is defined as:

**Inversion effect = Upright accuracy − Inverted accuracy**

|              Seed |            Upright |           Inverted | Inversion Effect |
| ----------------: | -----------------: | -----------------: | ---------------: |
|                 1 |             97.19% |             85.44% |   **+11.75 pts** |
|                 2 |             96.97% |             83.87% |   **+13.10 pts** |
|                 3 |             98.22% |             87.38% |   **+10.84 pts** |
|                 4 |             98.19% |             88.67% |    **+9.52 pts** |
|                 5 |             97.72% |             88.05% |    **+9.67 pts** |
| **Mean (95% CI)** | **97.66% (±0.71)** | **86.68% (±2.46)** |   **+10.98 pts** |

Each of the first four runs used **156,000 trials**. Seed 5 used **153,660 trials**.
