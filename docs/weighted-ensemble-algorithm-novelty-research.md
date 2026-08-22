# Does the validation-weighted ensemble count as a new algorithm?

> **Implementation update (22 August 2026):** The fixed global weighted-voting design
> evaluated in this note has been superseded on `main` by Adaptive
> Reliability-Uncertainty Fusion (ARUF). ARUF uses class-specific validation reliability,
> per-input entropy, and prediction agreement rather than one fixed model-weight vector.
> The earlier novelty conclusion still applies to fixed weighted soft voting. ARUF is
> described conservatively as a project-specific algorithm assembled from established
> ideas, not as a globally unprecedented learning method. See
> [the implementation plan](adaptive-fusion-implementation-plan.md) and
> [ADR 0002](adr/0002-add-adaptive-reliability-uncertainty-fusion.md).

**Research date:** 22 August 2026
**Short answer:** **No, not as currently defined.** It is a project-specific implementation of established weighted soft voting with validation-set weight tuning. The particular base-model combination, 0.05 search grid, macro-F1 objective, and deterministic tie-breaks may be original implementation choices, but they do not by themselves constitute a new general learning algorithm.

## What the repository implements

The implementation is on the unmerged `agent/weighted-hybrid-ensemble` branch (current tip `f74191c`), rather than on `main` (`8254643`). Its essential rule is

\[
\hat p(c\mid x)=\sum_{m=1}^{3}w_m\hat p_m(c\mid x),
\qquad
w_m>0,
\qquad
\sum_m w_m=1,
\]

followed by

\[
\hat y=\arg\max_c \hat p(c\mid x).
\]

The training code aligns each model's class-probability columns, exhaustively enumerates all positive three-weight combinations in increments of 0.05, and selects the candidate with the highest validation macro-F1. It breaks macro-F1 ties using multiclass log loss, distance from equal weights, and then the weight tuple itself ([repository training implementation, lines 80–163](https://github.com/ngyinhao/artificial-intelligence-asg/blob/f74191c9c47f707cb787a0d1ece8bc0467b4241d/complaint_compass/train.py#L80-L163)). With 20 grid units divided among three strictly positive weights, this evaluates \(\binom{19}{2}=171\) candidates.

At inference, the code obtains the three base models' probability matrices, aligns their class order, multiplies each by its saved weight, adds them, normalizes the result, and predicts with `argmax` ([repository inference implementation, lines 149–202](https://github.com/ngyinhao/artificial-intelligence-asg/blob/f74191c9c47f707cb787a0d1ece8bc0467b4241d/complaint_compass/inference.py#L149-L202)). The stored result is 0.05 for Naive Bayes, 0.70 for the calibrated Linear SVM, and 0.25 for MiniLM plus Logistic Regression ([ensemble metadata](https://github.com/ngyinhao/artificial-intelligence-asg/blob/f74191c9c47f707cb787a0d1ece8bc0467b4241d/artifacts/models/weighted_ensemble/metadata.json)).

This is best characterized as **global, validation-tuned, weighted soft voting** (or a convex linear probability pool). The weights are global because the same three values are used for every sample and class.

## Prior art

### 1. The prediction rule is standard weighted soft voting

Scikit-learn's official `VotingClassifier` documentation defines soft voting as predicting from the summed class probabilities and explicitly supports weights applied to class probabilities before averaging. Its `predict_proba` output is the weighted average probability for each class ([scikit-learn `VotingClassifier` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)). The official source performs `np.average(..., weights=...)`, while soft prediction takes the `argmax` ([scikit-learn source](https://github.com/scikit-learn/scikit-learn/blob/cc50648cc/sklearn/ensemble/_voting.py)). This is the same mathematical prediction rule used by the repository, notwithstanding its custom artifact and class-alignment code.

The method also predates scikit-learn. Kittler et al. discussed averages and other linear combinations of posterior-probability outputs and developed the sum rule within a common classifier-combination framework ([Kittler et al., 1998](https://doi.org/10.1109/34.667881); [author-hosted paper](https://cmp.felk.cvut.cz/~matas/papers/kittler-pami98.pdf)). Hashem and Schmeiser had already studied optimal linear combinations of trained model outputs ([Hashem & Schmeiser, 1995](https://doi.org/10.1109/72.377990)).

### 2. Learning ensemble weights from held-out performance is also established

Caruana et al. define an ensemble as predictions combined by weighted averaging or voting. Their ensemble-selection method repeatedly adds the model that maximizes performance on a validation ("hillclimb") set; selection with replacement gives repeatedly selected models greater weight. They explicitly allow optimization for different performance metrics and describe multiclass use through per-class predicted probabilities ([Caruana et al., 2004](https://doi.org/10.1145/1015330.1015432); [author-hosted corrected paper](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)). The repository uses exhaustive enumeration instead of forward selection, but the general idea—choose probability-averaging ensemble weights by held-out metric performance—is not new.

Large, Lines, and Bagnall present an especially close classifier-specific precedent: their cross-validation accuracy weighted probabilistic ensemble weights class-probability vectors using estimated base-classifier performance, sums and normalizes them, then predicts by `argmax` ([Large et al., 2019](https://doi.org/10.1007/s10618-019-00638-y); [open-access paper](https://ueaeprints.uea.ac.uk/id/eprint/71086/4/Published_Version.pdf)). Their formula for deriving weights differs from this repository's direct macro-F1 grid search, but the broader mechanism of validation-performance-weighted probability voting is already established.

The search procedure itself is an ordinary discrete hyperparameter search. Scikit-learn defines grid search as exhaustive search over specified parameter values, scored on held-out or cross-validation data ([official `GridSearchCV` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)). Restricting weights to a positive simplex and using a 0.05 grid changes the search space, not the basic kind of optimization.

## Assessment

| Question | Assessment |
|---|---|
| Is it an algorithm? | Yes. It is a deterministic training-and-inference procedure. |
| Is weighted soft voting new? | No. Probability averaging with classifier weights is established. |
| Is validation-based weight optimization new? | No. Held-out/CV performance has long been used to select ensemble members or weights. |
| Does macro-F1 plus the stated tie-break order make it new? | Not plausibly on the present evidence. These are objective and reproducibility choices within a standard grid-search framework. |
| What can the project claim? | A project-specific ensemble configuration, implementation, and empirical application to six-class complaint routing. |
| What should it avoid claiming? | “A new algorithm,” “a novel ensemble algorithm,” or an unprecedented way to learn soft-voting weights. |

Recommended report wording:

> We implemented a validation-tuned weighted soft-voting ensemble of three heterogeneous classifiers. Positive global weights were selected by exhaustive search on validation macro-F1, with log loss and deterministic criteria used to break ties. This is an application of established classifier-fusion and ensemble-selection principles, not a newly invented learning algorithm.

## What would be needed for a credible algorithmic-novelty claim?

A stronger claim would require a material methodological difference, not merely a different dataset, component list, metric, or grid resolution. Examples could include a genuinely new and justified weight-learning rule, class- or instance-conditional weighting, a new optimization formulation or constraint with demonstrated benefit, or a theoretical result. It would also require:

1. a systematic prior-art review showing how the method differs from the closest techniques;
2. a precise algorithm specification and complexity analysis;
3. ablations isolating the claimed innovation from ordinary soft voting and standard ensemble selection;
4. evaluation across multiple datasets or settings, with statistical uncertainty; and
5. an untouched evaluation protocol.

The repository itself notes an additional evidence limitation: the base-model test results had already been inspected before this ensemble was proposed, so the four-model test comparison is exploratory rather than confirmatory ([branch model card](https://github.com/ngyinhao/artificial-intelligence-asg/blob/f74191c9c47f707cb787a0d1ece8bc0467b4241d/docs/model-card.md)). This does not decide novelty, but it prevents the current test result from serving as strong evidence that the proposed configuration generalizes.

## Scope and caveat

This is a technical and academic novelty assessment, not a patentability opinion. Patent novelty is jurisdiction- and claim-specific and would require a formal prior-art search. The literature check here is sufficient to reject the broad academic claim that validation-weighted soft voting itself is new; it is not an exhaustive search of every possible detail of the repository's tie-breaking procedure.

## References

- Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. *Proceedings of the 21st International Conference on Machine Learning*. https://doi.org/10.1145/1015330.1015432
- Hashem, S., & Schmeiser, B. (1995). Improving model accuracy using optimal linear combinations of trained neural networks. *IEEE Transactions on Neural Networks, 6*(3), 792–794. https://doi.org/10.1109/72.377990
- Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 20*(3), 226–239. https://doi.org/10.1109/34.667881
- Large, J., Lines, J., & Bagnall, A. (2019). A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates. *Data Mining and Knowledge Discovery, 33*(6), 1674–1709. https://doi.org/10.1007/s10618-019-00638-y
- Scikit-learn developers. (2026). `GridSearchCV`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Scikit-learn developers. (2026). `VotingClassifier`. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
