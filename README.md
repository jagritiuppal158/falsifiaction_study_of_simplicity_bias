# falsifiaction_study_of_simplicity_bias
# Is It Simplicity or Speed? A Controlled Test of Simplicity Bias

A small pilot study for the Falsification Challenge: *do neural networks prefer simple features because they are simple, or because simple features are learned faster or have a larger margin?*

**Status: pilot.** 5 seeds, one data family, one network size. The results are exploratory and say nothing about simplicity bias in general.

## Question

When two features predict the same label and their learning speed is matched, does the complexity of a feature still decide which one a small neural network relies on?

## Method in brief

- **Data.** Each example has 20 numbers: feature A, feature B and 18 standard-normal distractors. Labels are -1 or +1 and balanced.
  - **A (simple):** the label times a random value in [0.5, 1.5]. One cut-off separates the classes.
  - **B (R regions):** R equal-width regions of width `s` alternate between the classes, with 15% left empty at each edge. R = 2 is a linear control; R = 3, 5, 7 are more complex.
  - Both features predict the label perfectly on their own.
- **Model.** MLP with 2 hidden layers of 64 ReLU units, binary cross-entropy, PyTorch default initialization.
- **Training.** Full batch, 2,000 steps, 1,000 training and 4,000 test points. Optimizers: Adam (lr 0.001) and gradient descent with momentum (lr 0.05, momentum 0.9, called "SGD" in the notebook; it is full batch, not stochastic).
- **Conditions.**
  - *Unmatched:* `s = 1`.
  - *Matched:* `s` is tuned by bisection on a log scale in [0.3, 30] (5 steps, 2 calibration seeds) until B alone reaches training loss 0.1 in as many steps as A alone.
- **Reliance measure.** Shuffle one feature across the test examples and record the accuracy lost. The **gap** is the drop when A is shuffled minus the drop when B is shuffled. About +0.5 means the network used only A, about 0 means it used both, and negative means it leaned on B.
- **Valid matched test.** Speed ratio (steps for B alone divided by steps for A alone) within 10% of 1, and test accuracy of at least 0.9.

## Main pilot findings

- Unmatched, the network used only A for every R and both optimizers (shuffling A cost 0.48 to 0.50 accuracy; shuffling B cost at most 0.02), even at R = 2 where both features are linear. That cannot be about complexity.
- Only 3 of 8 matched cells are valid tests (Adam R = 2, momentum-GD R = 2, momentum-GD R = 3). In those, the gap fell from about +0.5 to between -0.01 and +0.16.
- For R = 5 and 7, B could not be made as fast as A within the scale range, so those cells say nothing yet.
- Reading: the pilot supports the weaker claim that most of the unmatched preference comes from speed, margin or both. It cannot say whether a small direct effect of complexity remains.

## Files

```
.
├── README.md
├── proposal/
│   └── proposal.pdf              full research proposal
├── notebooks/
│   └── simplicity_bias.ipynb     the pilot experiment (PyTorch)
├── results/
│   ├── results.csv               one row per run
│   └── pilot_figure.png          bar plot of shuffle drops
└── requirements.txt
```

## How to run

1. Open `notebooks/simplicity_bias.ipynb` in Google Colab (or Jupyter).
2. Keep `PILOT = True` (the setting used for the reported results). It takes about 12 minutes on a CPU.
3. Run the cells in order.

`PILOT = False` uses a wider network (128 units), 2,000 training points, 2,500 steps and 7 calibration steps. It was **not** run for this pilot.

**Notebook cells, in order:** settings; data generator; model, training and reliance functions; speed matching; grid run (writes `results.csv`); summary table; pass/fail checks C1 to C5; bar plot (invalid matched cells are marked with a cross).

**Seeds.** Data seed is `1000 + seed`; network seed is `seed`; calibration runs use seeds 100 and 101.

**`results.csv` columns.** `opt`, `R`, `cond`, `scale`, `stepsA`, `stepsB`, `seed`, `acc`, `dA`, `dB` (accuracy drops when A and B are shuffled).

## Requirements

Python 3, `torch`, `numpy`, `pandas`, `matplotlib`. Install with `pip install -r requirements.txt`.

## Limitations

- Only 5 seeds, width 64 and 1,000 training points; no confidence intervals or equivalence test.
- Inputs are not standardized, and the scale `s` changes B's size and margin together, so size is not controlled.
- Speed matching can fail silently for large R, because speed does not always increase with scale (for Adam at R = 7, B alone needed about 206 steps at `s = 1` and about 658 at `s` of about 28).
- "SGD" is full-batch gradient descent with momentum, so mini-batch noise is not tested.
- The shuffle drop is only a stand-in for the planned conflict-set preference index.
- No training-dynamics checkpoints, so "learned first" cannot be separated from "relied on at the end".
- Slab count and frequency rise together, so the design cannot separate "fewer linear pieces" from "lower frequency".
- Synthetic data and a small MLP only. Nothing here transfers automatically to real images or larger models.

## Not run yet

Conflict-set preference index; learning-order checkpoints; probes for "learned but unused"; random-rotation control; parity and distributed feature families; exchange-rate grid; mini-batch SGD without momentum; the `PILOT = False` setting.

## Next steps

1. Standardize inputs and tune B's margin instead of its scale.
2. Add the conflict-set preference index.
3. Use 10 or more seeds with bootstrap intervals and an equivalence test (band of ±0.10).
4. Widen or replace the matching knob so R = 5 and 7 can be matched, or report that they cannot.
5. Record reliance every 50 steps.
6. Add mini-batch SGD and a second learning rate per optimizer.

## References

1. Shah et al. (2020). The Pitfalls of Simplicity Bias in Neural Networks. NeurIPS 2020.
2. Morwani et al. (2023). Simplicity Bias in 1-Hidden Layer Neural Networks. NeurIPS 2023.
3. Rahaman et al. (2019). On the Spectral Bias of Neural Networks. ICML 2019.
4. Kalimeris et al. (2019). SGD on Neural Networks Learns Functions of Increasing Complexity. NeurIPS 2019.
5. Pezeshki et al. (2021). Gradient Starvation: A Learning Proclivity in Neural Networks. NeurIPS 2021.
6. Hermann and Lampinen (2020). What Shapes Feature Representations? NeurIPS 2020.
7. Scimeca et al. (2022). Which Shortcut Cues Will DNNs Choose? ICLR 2022.
8. Puli et al. (2023). Don't Blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy. NeurIPS 2023.

The full reference list is in the proposal.
