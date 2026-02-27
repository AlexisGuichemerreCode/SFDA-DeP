# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

## Abstract

Domain adaptation transfers models across mismatched source and target distributions. In the source-free setting, SFDA adapts a source-trained model using only unlabeled target data, typically by relying on the model’s prediction distribution on the target domain to derive pseudo-labels or clusters. Under severe shifts, this distribution can become skewed and overconfident, collapsing to a few dominant classes and leaving others poorly represented, which further amplifies self-training and causes adaptation to collapse. In Weakly Supervised Object Localization (WSOL), the same skew propagates to spatial evidence through class-conditioned CAMs, degrading localization. To address this issue, we propose Debias-then-Adapt (DA-SFDA), a SFDA method for WSOL that unlearns before adaptation. At first, it performs a target-guided unlearning to partition target samples into retain/forget sets and suppress biased target predictions. To preserve localization during debiasing, a lightweight pixel-level head is jointly trained using CAM-derived foreground/background priors, stabilizing spatial evidence. Then, the resulting debiased model enables downstream SFDA with more balanced and reliable pseudo-labels.
### Issues:
Please create a github issue.
