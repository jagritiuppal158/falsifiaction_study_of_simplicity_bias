# falsifiaction_study_of_simplicity_bias
# Is It Simplicity or Speed? A Controlled Test of Simplicity Bias

A small pilot for the Falsification Challenge. When two features predict the same label, does a neural network prefer the simple one because it is **simple**, or because it is **learned faster** or has a **larger margin**?

> **Status: pilot.** 5 seeds, one data family, one small network. Results are exploratory and say nothing about simplicity bias in general.

## Key result

| Case (valid matched tests only) | Gap unmatched | Gap matched |
|---|---|---|
| Adam, R = 2 | +0.47 | 0.00 |
| Momentum GD, R = 2 | +0.49 | +0.16 |
| Momentum GD, R = 3 | +0.50 | -0.01 |

*Gap = accuracy lost when A is shuffled minus accuracy lost when B is shuffled. About +0.5 means the network used only A, about 0 means it used both, negative means it leaned on B.*

- **Unmatched**, the network used only the simple feature A in every case, even at R = 2 where B is also linear. That preference cannot be about complexity.
- **Speed-matched**, the gap fell from about +0.5 to between -0.01 and +0.16.
- **Only 3 of 8 matched cells are valid tests.** For R = 5 and 7, B could not be made as fast as A within the scale range, so those cells say nothing.
- **Reading:** most of the unmatched preference comes from speed, margin or both. The pilot cannot say whether a small direct effect of complexity remains.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── simplicity_bias.ipynb      the pilot experiment (PyTorch)
└── Is It Simplicity or Speed A Controlled Test of Simplicity Bias.docx
                               full research proposal
```

Running the notebook also writes `results.csv` (one row per run). It is not stored in the repository until you commit it.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook simplicity_bias.ipynb      # or open it in Google Colab
```

Run all cells in order. With `PILOT = True` (the setting used for the reported results) the full grid takes about 12 minutes on a CPU.

`PILOT = False` uses a wider network (128 units), 2,000 training points, 2,500 steps and 7 calibration steps. It was **not** run for this pilot.

## Method

**Data.** Each example has 20 numbers: feature A, feature B and 18 standard-normal distractors. Labels are -1 or +1, balanced. Both features predict the label perfectly alone.
- **A (simple):** label x uniform(0.5, 1.5). One cut-off separates the classes. Margin 0.5.
- **B (R regions):** R regions of width `s` alternate between the classes, with 15% left empty at each edge, so B's margin is 0.15 x `s`. R = 2 is a linear control; R = 3, 5, 7 are more complex.

**Model and training.** MLP with 2 hidden layers of 64 ReLU units, binary cross-entropy, PyTorch default initialization. Full batch, 2,000 steps, 1,000 training and 4,000 test points. Optimizers: Adam (lr 0.001) and gradient descent with momentum (lr 0.05, momentum 0.9). The notebook calls the second one `sgd`, but it is full batch, not stochastic.

**Speed matching.**
- Speed = first step at which training loss falls below 0.1 (2,000 if never).
- *Unmatched:* `s = 1`.
- *Matched:* `s` is tuned by bisection on a log scale in [0.3, 30] (5 steps, 2 calibration seeds) until B alone reaches the threshold in as many steps as A alone.
- A matched cell is a **valid test** only if the speed ratio (steps for B divided by steps for A) is within 10% of 1 and test accuracy is at least 0.9.

**Reliance measure.** Shuffle one feature across the test examples and record the accuracy lost (`dA`, `dB`). This is a stand-in for the planned conflict-set preference index (see below).

## Reading the results

`results.csv` columns: `opt`, `R`, `cond` (unmatched or matched), `scale` (`s`), `stepsA`, `stepsB`, `seed`, `acc`, `dA`, `dB`.

The notebook's summary table adds `speed_ratio = stepsB / stepsA`. Checks C1 to C5 are pass/fail tests written before reading the results: C1, C2, C4 and C5 pass; **C3 fails** (matched R = 2 moved the network toward balance, not toward B). The bar plot marks invalid matched cells with a cross.

**Seeds.** Data seed is `1000 + seed`; network seed is `seed`; calibration uses seeds 100 and 101.

### Preference index (planned, not yet run)

Test on a *conflict set* where A and B point to opposite labels. PI = (fraction of predictions that follow A) - 0.5. PI = +0.5 means pure reliance on A, 0 means no preference, -0.5 means pure reliance on B. The H0 equivalence band is +/-0.10.

## Limitations

- 5 seeds, width 64, 1,000 training points; no confidence intervals or equivalence test.
- Inputs are not standardized, and `s` changes B's size and margin together, so size is not controlled.
- Speed matching can fail silently: speed does not always increase with `s` (Adam, R = 7: about 206 steps at `s = 1`, about 658 at `s` of about 28).
- "SGD" is full-batch gradient descent with momentum, so mini-batch noise is not tested.
- The shuffle gap is measured on shuffled, off-distribution inputs, and is not the conflict-set index.
- Only the final model is measured, so "learned first" cannot be separated from "relied on at the end".
- Slab count and frequency rise together, so "fewer linear pieces" and "lower frequency" are not separated.
- Synthetic data and a small MLP only. Nothing here transfers automatically to images or large models.

## Not run yet

Conflict-set preference index; reliance checkpoints during training; probes for "learned but unused"; random-rotation control; parity and distributed feature families; strength-versus-complexity grid; mini-batch SGD without momentum; the `PILOT = False` setting.

## Next steps

1. Standardize inputs and tune B's margin instead of its scale.
2. Add the conflict-set preference index.
3. Use 10 or more seeds with bootstrap intervals and an equivalence test (+/-0.10).
4. Widen or replace the matching knob so R = 5 and 7 can be matched, or report that they cannot.
5. Record reliance every 50 steps.
6. Add mini-batch SGD without momentum and a second learning rate per optimizer.

## Key references

1. Shah et al. (2020). The Pitfalls of Simplicity Bias in Neural Networks. NeurIPS 2020.
2. Morwani et al. (2023). Simplicity Bias in 1-Hidden Layer Neural Networks. NeurIPS 2023.
3. Rahaman et al. (2019). On the Spectral Bias of Neural Networks. ICML 2019.
4. Kalimeris et al. (2019). SGD on Neural Networks Learns Functions of Increasing Complexity. NeurIPS 2019.
5. Pezeshki et al. (2021). Gradient Starvation: A Learning Proclivity in Neural Networks. NeurIPS 2021.
6. Puli et al. (2023). Don't Blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy. NeurIPS 2023.

The full reference list is in the proposal.
