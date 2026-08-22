![Tunku Abdul Rahman University of Management and Technology logo](ComplaintCompass_Report_media/media/image1.png)

# Assignment Documentation

**Session:** 202605, Year 2026/27

**Project title:** ComplaintCompass: Comparative NLP for Automated Consumer Financial Complaint Routing

**Programme:** [PROGRAMME]

**Tutorial group:** [TUTORIAL GROUP]
**Tutor:** [TUTOR]

## Team members

| No. | Student name | Student ID | Module / contribution | Signature and date |
|---:|---|---|---|---|
| 1 | [STUDENT NAME] | [STUDENT ID] | [MODULE / CONTRIBUTION] | [SIGNATURE / DATE] |
| 2 | [STUDENT NAME] | [STUDENT ID] | [MODULE / CONTRIBUTION] | [SIGNATURE / DATE] |
| 3 | [STUDENT NAME] | [STUDENT ID] | [MODULE / CONTRIBUTION] | [SIGNATURE / DATE] |

# Introduction

## Background

Consumer complaint narratives are unstructured accounts that must be converted into actionable routing information before downstream review can begin. Natural language processing (NLP) frames this requirement as supervised multiclass text classification: a model receives a narrative and assigns one label from a predefined product taxonomy. The Consumer Financial Protection Bureau (CFPB) makes complaint data available for public use and notes that published narratives are consumers' own descriptions, shared only when consumers opt in and after steps to remove personal information (Consumer Financial Protection Bureau \[CFPB\], 2026). These characteristics make the database useful for an academic routing study while also requiring caution about privacy, representativeness, and verification.

ComplaintCompass is an English-language academic prototype that compares complementary representations under one deterministic evaluation protocol. The base approaches are term frequency-inverse document frequency (TF-IDF) word unigrams and bigrams with Multinomial Naive Bayes; the same sparse representation with a calibrated Linear Support Vector Machine (SVM); and all-MiniLM-L6-v2 sentence embeddings with Logistic Regression. A validation-weighted soft-voting ensemble combines those three probability vectors using fixed global weights. Adaptive Reliability-Uncertainty Fusion (ARUF) instead adapts their influence using validation-derived class reliability, per-input uncertainty, and prediction agreement. The application is intended to assist an analyst or evaluator by showing a predicted category, calibrated confidence, and three leading candidates. It does not judge complaint merit, determine legal or financial outcomes, or replace accountable human review.

The central argument is that useful routing accuracy is achievable on the six selected CFPB products, but a higher score is not the only decision criterion. Predictive performance must be read with latency, artifact size, calibration, evidence history, and application boundaries. The weighted ensemble provides the strongest generated validation macro-F1 (0.8425) and exploratory test macro-F1 (0.8513), so it is the registered default. ARUF achieved validation macro-F1 0.8263 and exploratory test macro-F1 0.8363: it improved on MiniLM Logistic Regression and Naive Bayes but did not exceed Linear SVM or the fixed ensemble. This negative comparison is retained because implementing a new algorithm does not guarantee superior empirical performance.

## Problem Statement

The immediate problem is to assign complaint narratives to the appropriate product category. Manual triage can be slow and inconsistent because narratives vary in length, terminology, and clarity, and a single narrative may refer to more than one financial product. Incorrect routing can delay handling and reduce the usefulness of complaint-management workflows. The investigation therefore asks whether sparse lexical features, semantic sentence embeddings, and probability combination can classify a balanced six-product subset reliably, and which approach offers the best balance of macro-F1, latency, size, transparency, and evidential strength. No claim is made about CFPB staffing levels, monetary savings, or realized production impact.

## Objectives/Aims

1. Build a reproducible, balanced six-class dataset from public CFPB complaint narratives.

2. Implement and tune three complementary NLP classifiers under a common data split and evaluation protocol.

3. Implement and assess ARUF as a validation-tuned, adaptive probability-fusion extension, while treating its eventual test result as exploratory because the original holdout had already been inspected.

4. Compare models using macro-F1 as the primary measure, supported by accuracy, macro precision, macro recall, weighted F1, per-class results, latency, and artifact size.

5. Develop a Streamlit prototype that returns a predicted category, calibrated confidence, and top-three candidates without intentionally storing submitted text.

6. Analyse errors, limitations, ethical risks, and model suitability for the bounded academic routing scenario.

## Significance / Contribution of the Study

Practically, the project supplies a functioning demonstration of complaint routing and a transparent model-comparison view. Methodologically, it places sparse lexical and transformer-derived semantic representations on the same processed records and deterministic split, avoiding comparisons confounded by different samples or metrics. ARUF is a project-specific composition of established reliability weighting, entropy-based confidence, and classifier-agreement concepts; it is not presented as a globally novel learning algorithm. Educationally, the project records provenance through download and processed-data manifests, a dataset checksum, saved metadata, a registry, command-line workflows, and automated test sources. The contribution is the controlled comparison, reproducible implementation, application integration, and critical interpretation of trade-offs.

# Related Work

## Review of previous studies

Automated complaint classification sits within the broader field of supervised text categorization. Joachims (1998) showed why maximum-margin methods are well suited to sparse, high-dimensional text, while McCallum and Nigam (1998) compared event models for Naive Bayes and found the multinomial formulation particularly effective with larger vocabularies. These foundations motivate the two TF-IDF pipelines in ComplaintCompass: Naive Bayes as a computationally economical probabilistic baseline and Linear SVM as a stronger margin-based classifier. Their inclusion also provides a meaningful test of whether a modern sentence representation is necessary for this bounded taxonomy.

Financial complaint research has increasingly treated narrative text as a classification resource. Miranda, Kohnecke, and Renard (2023) used CFPB narratives in a hierarchical-classification benchmark, predicting product and sub-product labels and showing that taxonomy structure can matter. More recent bilingual work by Jain et al. (2026) compared TF-IDF approaches with a fine-tuned multilingual transformer on a balanced synthetic English-Hindi dataset and reported higher accuracy for the contextual model. The studies are not directly comparable to ComplaintCompass because their label structures, language conditions, sampling, representations, and evaluation designs differ. They nevertheless support the use of complaint narratives as predictive input and the need to compare feature-based and contextual approaches under controlled conditions.

Sentence-transformer models address a different representation question. Reimers and Gurevych (2019) introduced Sentence-BERT to produce semantically meaningful fixed-size sentence embeddings more efficiently than pairwise BERT inference. ComplaintCompass uses the compact all-MiniLM-L6-v2 encoder to represent each narrative, then learns a Logistic Regression decision boundary. This separates representation learning from task classification and offers a semantic alternative to exact lexical matching. However, pretrained semantic representations do not guarantee superior downstream performance: domain vocabulary, truncation, class boundaries, and the strength of a tuned sparse baseline all affect the result.

Probability calibration and combination are needed when scores must be presented as confidence or combined across models. Maximum-margin outputs are not probabilities; cross-validated calibration maps decision values to the unit interval using predictions from data not used to fit each estimator (scikit-learn developers, 2026a). Standard weighted soft voting forms a linear pool of aligned class-probability vectors (Kittler et al., 1998), while validation-based ensemble selection is also established (Caruana et al., 2004; Large et al., 2019). ARUF extends this project beyond fixed global weights by adapting member influence using category-specific validation F1, entropy-derived confidence for the current input, and model agreement. These ingredients are established ideas assembled for this application, not evidence of an unprecedented algorithm. The methodological advantage remains conditional on strict separation of tuning and final evaluation; because the original test results had already been inspected before ARUF was proposed, its eventual result on that test set must be treated as exploratory.

Evaluation must make class treatment explicit. Accuracy summarizes the proportion of correct decisions but can hide uneven class behavior. Macro averaging computes the arithmetic mean of per-class scores, giving each product equal weight, whereas weighted F1 scales each class by its support (scikit-learn developers, 2026b). Because this dataset is deliberately balanced, accuracy and macro recall are numerically close, yet macro-F1 remains the primary measure because it combines precision and recall at class level. Confusion matrices, per-class scores, inference latency, and stored size add diagnostic and practical evidence that a single aggregate score cannot provide.

## Research gap and justification for the current study

Prior work demonstrates both classical and contextual complaint classification, but isolated headline scores do not reveal the operational trade-offs among sparse models, fixed sentence embeddings, and adaptive probability fusion under the same records, split, metrics, and application interface. Furthermore, aggregate accuracy can conceal category-specific errors, while research prototypes may omit serving latency, artifact size, provenance, or privacy behavior. ComplaintCompass addresses this bounded gap through a reproducible side-by-side comparison tied to a working prototype. It does not claim that these methods or ARUF's constituent ideas have never been used; rather, it asks which trade-off is defensible for this specific six-class, English-only CFPB routing scenario.

# Methodology

## System flowchart / activity diagram

![ComplaintCompass workflow covering data preparation, model development, evaluation, and application inference.](ComplaintCompass_Report_media/media/image2.png)

*Figure 1. ComplaintCompass training, evaluation, and inference workflow.*

Data preparation first attempted bounded requests to the official CFPB source. If the API route failed, deterministic byte ranges from the official CSV export served as a fallback rather than a second dataset. The pipeline retained complaint ID, date received, narrative, and product; normalized HTML, Unicode, whitespace, and redaction markers; removed missing, short, exact-duplicate, and conflicting-label narratives; truncated text to 2,000 characters; then sampled 3,000 records per class with seed 42. The processed data manifest records the 70/15/15 stratified split and checksum.

Model development used five-fold stratified cross-validation on training data for model-specific tuning. The three candidates were compared on the validation set, after which final base models were refitted on training plus validation data and saved with labels and metadata. Both combination methods use validation-stage probability outputs generated by base models fitted on training data only. The weighted ensemble selected fixed weights of 0.05, 0.70, and 0.25 for Naive Bayes, Linear SVM, and MiniLM respectively. ARUF derives per-model, per-class reliability from validation F1 and selected $\alpha=1.0$, $\beta=0.5$, and $\gamma=0.2$. The saved configurations, registry entries, inference paths, application selector, metrics, error samples, and confusion matrices were regenerated through the shared pipeline.

At inference, an English narrative must contain 20 to 2,000 valid characters. The text is normalized in memory, a registered saved model is loaded, and its six-class probability vector is used to display the predicted category, confidence, and top three candidates. The validation-weighted ensemble is the default. The application code does not intentionally log or persist submitted narratives, although real deployment would still require independent security, telemetry, and privacy review.

> **Methodological note.** The original base-model test results had already been inspected before the combination analysis was completed. Although both combination methods use validation data rather than test labels, the five-model comparison on the existing test split is post-hoc and therefore exploratory rather than confirmatory. A later untouched or time-based holdout is required for confirmation.

## Description and analysis of dataset

The source is the official CFPB Consumer Complaint Database. The extraction period runs from 1 August 2023 through 31 December 2025, aligning with the current product taxonomy recorded in the project documentation. Four fields were retained: complaint ID, date received, consumer complaint narrative, and product. The six target products are shown in Table 1. Each contains 3,000 sampled records, so every class contributes 2,100 training, 450 validation, and 450 test observations.

| **Product label** | **Train** | **Validation** | **Test** | **Total** |
|----|----|----|----|----|
| Checking or savings account | 2,100 | 450 | 450 | 3,000 |
| Credit card | 2,100 | 450 | 450 | 3,000 |
| Credit reporting or other personal consumer reports | 2,100 | 450 | 450 | 3,000 |
| Debt collection | 2,100 | 450 | 450 | 3,000 |
| Money transfer, virtual currency, or money service | 2,100 | 450 | 450 | 3,000 |
| Mortgage | 2,100 | 450 | 450 | 3,000 |

*Table 1. Six product labels and deterministic split counts.*

Cleaning normalized HTML fragments, Unicode, whitespace, and CFPB redaction markers; rejected narratives shorter than 20 characters; limited accepted text to 2,000 characters; removed exact duplicates; and removed identical narratives associated with conflicting labels. Deterministic balanced sampling used seed 42. The final dataset contains 18,000 rows: 12,600 training, 2,700 validation, and 2,700 test. Its processed SHA-256 checksum is 8727affded9c28e0e3886264c34a53a8a7705d81216bd5d744f943a6cf3d7719.

The dataset is not a statistical sample of all consumers. CFPB explains that published narratives are opt-in, complaint experiences are not independently verified, and complaint volume should not be interpreted without context (CFPB, 2026). The sample is US-specific and English-only, reflects a selected time period and six-class subset, may contain residual privacy-sensitive context, and truncates long narratives. Balancing supports a fair class comparison but changes natural prevalence. These constraints limit external validity and prohibit interpreting the classifier as a measure of complaint merit, legal violation, or population harm.

## Algorithm selection & description of algorithms

| **Model** | **Representation / classifier** | **Tuning** | **Rationale and limitation** |
|----|----|----|----|
| Multinomial Naive Bayes | Word TF-IDF, 1-2 grams; `MultinomialNB` | $\alpha \in \{0.1, 0.5, 1.0\}$; selected $0.1$ | Fast, small probabilistic baseline; conditional-independence assumptions can produce uneven recall. |
| Calibrated Linear SVM | Word TF-IDF, 1-2 grams; `LinearSVC` + cross-validated calibration | $C \in \{0.5, 1.0, 2.0\}$; selected $0.5$ | Strong sparse-text margin classifier with probabilities; depends heavily on lexical evidence. |
| MiniLM + Logistic Regression | all-MiniLM-L6-v2 sentence embedding; Logistic Regression | $C \in \{0.5, 1.0, 2.0\}$; selected $2.0$ | Compact semantic representation; slower and larger, and generic semantics may not match product boundaries. |
| Weighted ensemble | Aligned probabilities from all three base models | Positive weights on a 0.05 grid; selected 0.05, 0.70, and 0.25 | Transparent fixed global combination; requires all member artifacts. |
| ARUF | Aligned probabilities from all three base models | $\alpha,\beta \in \{0.5,1,2\}$ and $\gamma \in \{0,0.05,0.10,0.20\}$; selected $\alpha=1.0$, $\beta=0.5$, $\gamma=0.2$ | Adapts model influence by class reliability, current uncertainty, and agreement; requires all member artifacts. |

*Table 2. Model representations, classifiers, tuned parameters, and roles.*

Five-fold stratified cross-validation preserved class proportions within each fold. Hyperparameters were selected using training/CV evidence; each selected candidate was then evaluated on validation data. The fixed ensemble weights and ARUF parameters were selected using the three validation probability matrices without consulting test labels. Final base models were refitted on training plus validation data before the test benchmark. The model with the highest validation macro-F1 is the application default; size and latency break exact ties only. The weighted ensemble therefore becomes the default.

### Adaptive Reliability-Uncertainty Fusion

Let $M$ be the number of member models, $K$ the number of classes, and $p_{m,k}(x)$ the calibrated probability assigned by model $m$ to class $k$ for input $x$. All member probability columns are first aligned to the same fixed class order and normalized so that

$$
p_{m,k}(x) \ge 0,
\qquad
\sum_{k=1}^{K} p_{m,k}(x) = 1.
$$

The reliability of member $m$ for class $k$ is its validation-set class F1 score:

$$
r_{m,k}=F^{(\mathrm{val})}_{1,m,k}.
$$

ARUF measures a member's uncertainty for the current input with normalized entropy:

$$
H_m(x)
=
-\frac{1}{\log K}
\sum_{k=1}^{K} p_{m,k}(x)\log p_{m,k}(x),
$$

and converts it to confidence:

$$
q_m(x)=\max\!\left(\varepsilon,\,1-H_m(x)\right),
$$

where $\varepsilon=10^{-12}$ prevents zero-valued weights. The adaptive contribution weight is

$$
w_{m,k}(x)
=
\max(\varepsilon,r_{m,k})^{\alpha}
q_m(x)^{\beta}.
$$

Each model also casts one hard vote. For class $k$,

$$
v_k(x)
=
\sum_{m=1}^{M}
\mathbf{1}\!\left[
\operatorname*{arg\,max}_{j} p_{m,j}(x)=k
\right],
$$

and the agreement multiplier is

$$
a_k(x)
=
1+\gamma\max\!\left(v_k(x)-1,\,0\right).
$$

The unnormalized class score and final fused probability are

$$
s_k(x)
=
a_k(x)
\sum_{m=1}^{M} w_{m,k}(x)p_{m,k}(x),
$$

$$
\widehat{p}_k(x)
=
\frac{s_k(x)}{\sum_{j=1}^{K}s_j(x)},
\qquad
\widehat{y}(x)
=
\operatorname*{arg\,max}_{k}\widehat{p}_k(x).
$$

The parameters $\alpha$, $\beta$, and $\gamma$ are selected deterministically using validation predictions only. Candidates are ranked first by descending validation macro-F1, then by ascending multiclass log loss, distance from the neutral values $(\alpha,\beta)=(1,1)$, $\gamma$, $\alpha$, and $\beta$. Log loss is computed with an explicit class-to-column mapping so the fixed domain label order cannot be silently rearranged. ARUF is a project-specific fusion rule built from established ideas; the report does not claim global algorithmic novelty.

## Evaluation metrics

For class $k$, true positives $TP_k$ are correct assignments to $k$, false positives $FP_k$ are other complaints incorrectly assigned to $k$, and false negatives $FN_k$ are complaints of $k$ assigned elsewhere. With $N$ samples, $K=6$ classes, and indicator $\mathbf{1}[\cdot]$, the core measures are

$$
\operatorname{Precision}_k
=
\frac{TP_k}{TP_k+FP_k},
$$

$$
\operatorname{Recall}_k
=
\frac{TP_k}{TP_k+FN_k},
$$

$$
F_{1,k}
=
2\frac{\operatorname{Precision}_k\operatorname{Recall}_k}
{\operatorname{Precision}_k+\operatorname{Recall}_k},
$$

$$
\operatorname{MacroF1}
=
\frac{1}{K}\sum_{k=1}^{K}F_{1,k},
$$

and

$$
\operatorname{Accuracy}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\!\left[y_i=\widehat{y}_i\right].
$$

Macro precision and macro recall average their per-class values equally; weighted F1 averages $F_{1,k}$ in proportion to class support. A confusion matrix records counts for every true/predicted class pair. Cross-validation mean and standard deviation describe training-stage stability. Mean per-item inference latency and artifact size characterize practical cost. Multiclass log loss is

$$
\mathcal{L}_{\mathrm{log}}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log\widehat{p}_{y_i}(x_i),
$$

which penalizes probability assigned away from the true class and serves as ARUF's first tie-breaker after macro-F1. Validation and test samples each contain 2,700 observations; reported scores use four decimal places and latency uses two.

# Result & Discussion

## Results

Table 3 presents the committed training-stage and validation evidence. Linear SVM achieved the highest base-model cross-validation score, while the fixed weighted ensemble achieved the highest validation macro-F1. The validation scores provide the cleanest pre-test basis for choosing the application default.

| **Model** | **Selected parameter** | **CV macro-F1 mean** | **CV SD** | **Validation macro-F1** | **Validation accuracy** |
|----|----|----|----|----|----|
| Weighted ensemble | NB = 0.05; SVM = 0.70; MiniLM = 0.25 | Not applicable | Not applicable | 0.8425 | 0.8422 |
| Linear SVM | C = 0.5 | 0.8366 | 0.0074 | 0.8353 | 0.8352 |
| ARUF | alpha = 1.0; beta = 0.5; gamma = 0.2 | Not applicable | Not applicable | 0.8263 | 0.8263 |
| MiniLM + LR | C = 2.0 | 0.8120 | 0.0080 | 0.8022 | 0.8022 |
| Naive Bayes | alpha = 0.1 | 0.7532 | 0.0061 | 0.7395 | 0.7444 |

*Table 3. Five-fold cross-validation and validation-set results (n = 2,700 validation records).*

Table 4 reports results backed by generated repository artifacts. The fixed weighted ensemble and ARUF are retained as distinct comparators: one uses global validation-selected weights, while the other adapts influence by class and input. Latency was regenerated in the current environment and is therefore compared only within this measurement run.

| **Model / evidence status** | **Accuracy** | **Macro precision** | **Macro recall** | **Macro-F1** | **Latency (ms)** | **Size (MiB)** |
|----|----|----|----|----|----|----|
| Weighted ensemble - generated, post-hoc exploratory | 0.8511 | 0.8532 | 0.8511 | 0.8513 | 20.71 | 120.67 |
| Linear SVM - generated | 0.8456 | 0.8474 | 0.8456 | 0.8458 | 2.22 | 25.46 |
| ARUF - generated, post-hoc exploratory | 0.8363 | 0.8418 | 0.8363 | 0.8363 | 24.45 | 120.67 |
| MiniLM + LR - generated | 0.8148 | 0.8169 | 0.8148 | 0.8147 | 18.09 | 87.37 |
| Naive Bayes - generated | 0.7581 | 0.7911 | 0.7581 | 0.7524 | 0.51 | 7.84 |

*Table 4. Generated five-method test benchmark ($n=2{,}700$). Combination-method results are post-hoc exploratory.*

All five generated test macro-F1 values meet the project target of at least 0.75. This threshold is an internal success criterion, not evidence of real-world fitness. Figures 2-6 show the five confusion matrices generated by the shared evaluation pipeline.

![Weighted ensemble confusion matrix for six complaint product classes.](ComplaintCompass_Report_media/media/image7.png)

*Figure 2. Weighted ensemble confusion matrix on the exploratory test set.*

![Linear SVM confusion matrix for six complaint product classes.](ComplaintCompass_Report_media/media/image3.png)

*Figure 3. Linear SVM confusion matrix on the exploratory test set.*

![MiniLM with Logistic Regression confusion matrix for six complaint product classes.](ComplaintCompass_Report_media/media/image4.png)

*Figure 4. MiniLM + Logistic Regression confusion matrix on the exploratory test set.*

![Multinomial Naive Bayes confusion matrix for six complaint product classes.](ComplaintCompass_Report_media/media/image5.png)

*Figure 5. Naive Bayes confusion matrix on the exploratory test set.*

![ARUF confusion matrix for six complaint product classes.](ComplaintCompass_Report_media/media/image6.png)

*Figure 6. ARUF confusion matrix on the exploratory test set.*

| **Weighted-ensemble product** | **Precision** | **Recall** | **F1** | **Support** |
|----|----|----|----|----|
| Checking or savings account | 0.7738 | 0.8667 | 0.8176 | 450 |
| Credit card | 0.8470 | 0.8244 | 0.8356 | 450 |
| Credit reporting or other personal consumer reports | 0.8215 | 0.8489 | 0.8350 | 450 |
| Debt collection | 0.8376 | 0.8022 | 0.8195 | 450 |
| Money transfer, virtual currency, or money service | 0.8878 | 0.8089 | 0.8465 | 450 |
| Mortgage | 0.9513 | 0.9556 | 0.9534 | 450 |

*Table 5. Per-class performance of the default weighted ensemble on the exploratory test set.*

## Discussion/Interpretation

The first objective was achieved: the manifests identify a balanced, deterministic 18,000-record dataset with a stable checksum and 70/15/15 split. The second objective was achieved for the three base methods, which share one preprocessing and evaluation protocol. The third objective was achieved at prototype level: ARUF has deterministic fitting, class-order-safe log loss, a saved real-data configuration, registry and inference integration, automated coverage, and generated validation and exploratory test evidence. Confirmation on a fresh holdout remains future work.

ARUF exploited complementary lexical and semantic signals without assigning one fixed global weight to each model. Category reliability emphasized members where validation F1 was stronger, entropy reduced the influence of uncertain predictions, and the agreement multiplier reinforced classes selected by multiple members. ARUF improved macro-F1 over MiniLM by 0.0217 and over Naive Bayes by 0.0839, but remained 0.0095 below Linear SVM. The result shows that adaptive fusion produced a competitive compromise but could not recover enough complementary correct decisions to surpass the strongest sparse model.

The weighted ensemble is the application default because it achieved the highest validation macro-F1 (0.8425). It also achieved the highest exploratory test macro-F1 (0.8513), with 20.71 ms latency and a 120.67 MiB effective artifact footprint. Linear SVM is the efficient alternative at 0.8458 macro-F1, 2.22 ms, and 25.46 MiB. ARUF uses the same three members but reached 0.8363 macro-F1 at 24.45 ms, so its adaptive mechanism did not outperform the simpler fixed weighting.

Naive Bayes provides the smallest and fastest option at 7.84 MiB and 0.50 ms, but its 0.7524 macro-F1 is materially lower. The confusion matrix shows highly uneven behaviour: mortgage recall is 0.9489, whereas money-transfer recall is 0.4578 despite very high precision. The model is consequently useful as a baseline or constrained-device option, not the strongest general router. Its conditional-independence assumptions and reliance on token frequency provide a plausible explanation for overconfident, uneven class boundaries.

MiniLM's semantic representation did not outperform the tuned sparse Linear SVM. It achieved 0.8147 macro-F1 while requiring 87.37 MiB and 18.23 ms per item. This does not show that semantic embeddings are generally inferior; it shows that this compact generic encoder plus Logistic Regression was less effective on the selected labels and sample. Explicit product terminology and stable domain vocabulary may favour sparse word n-grams, while a fixed embedding can compress distinctions that matter for adjacent financial categories. Fine-tuning, domain-adapted encoders, or longer-context strategies remain open tests.

The Linear SVM confusion matrix shows mortgage as the strongest class, with F1 = 0.9488. A plausible explanation is that mortgage narratives contain distinctive terms such as escrow, foreclosure, and loan servicing. Checking/savings and credit card form a recurring confusion pair: 51 checking/savings complaints were predicted as credit card, and 54 credit-card complaints were predicted as checking/savings. Shared language about transactions, fees, fraud, accounts, and disputed charges can blur the boundary. Money-transfer and mortgage narratives also show directional confusion, potentially because narratives mention payments or servicing across products. These explanations are hypotheses; masking explicit product terms and evaluating deliberately ambiguous synthetic cases would test them.

The fourth objective was met for all five registered methods through macro and per-class metrics, confusion matrices, latency, artifact size, and error samples. The fifth objective was met for the five-method Streamlit interface, which validates input and presents prediction, confidence, alternatives, and model comparison while defaulting to the highest-validation-macro-F1 model. The sixth objective is addressed through the documented limitations and privacy boundaries. These achievements demonstrate technically useful routing in the studied setting, not production readiness or causal operational benefit.

# Conclusion

## Achievements

ComplaintCompass created a balanced six-class corpus of 18,000 public CFPB narratives with deterministic sampling, a recorded seed, a stratified split, manifests, and a processed checksum. This satisfies the reproducible-dataset objective within the availability and selection limits of the source.

Three complementary base approaches were implemented, trained, and evaluated under one protocol: Multinomial Naive Bayes, calibrated Linear SVM, and MiniLM embeddings with Logistic Regression. ARUF was also implemented, fitted, registered, and evaluated as an adaptive fusion of their aligned probability outputs. Its selected configuration and reliability matrix are preserved as a reproducible artifact.

All five evidenced methods met the internal 0.75 macro-F1 target on the exploratory benchmark. The weighted ensemble achieved the strongest generated result (0.8513 macro-F1) and is the evidence-selected default. Linear SVM followed at 0.8458 with much lower operational cost, while ARUF reached 0.8363, demonstrating that the proposed adaptive mechanism is competitive but not automatically superior to simpler fixed weighting or a strong tuned base model.

The Streamlit prototype loads saved models, validates a 20-2,000-character English input, and returns a category, confidence, and top-three candidates. Reproducibility and privacy controls include model metadata, a registry, manifests, generated metric files, test sources, local inference, and an explicit non-persistence design. Automated tests cover the fusion mathematics, deterministic selection, training and artifact reload, inference integration, and successful application classification.

Overall, the project demonstrates useful automated routing performance for the selected CFPB setting. The validation-weighted ensemble is the default because it has the highest validation macro-F1 and the strongest exploratory test result. Calibrated Linear SVM remains the substantially faster and smaller alternative. ARUF is a completed project-specific algorithm, but its post-hoc result still requires confirmation on a later untouched holdout.

## Limitations and Future Works

| **Limitation** | **Concrete future work** |
|----|----|
| US-specific, opt-in CFPB narratives | Validate on another complaint source and a later time period. |
| English-only model | Add language detection and independently evaluated multilingual models. |
| Six-class flat subset | Expand the taxonomy and evaluate hierarchical classification. |
| Post-hoc ARUF evaluation | Confirm the completed algorithm on a fresh untouched or time-based holdout. |
| Aggregate metrics may hide drift | Add time-sliced monitoring and carefully justified subgroup-safe analyses. |
| Possible reliance on explicit product words | Run masking/ablation tests and ambiguous-text evaluation. |
| Fixed 2,000-character truncation | Compare head-tail, salient-span, and long-context strategies. |
| No production validation | Conduct human-in-the-loop usability, calibration, robustness, security, privacy, and drift studies. |

*Table 6. Limitations paired with proposed future work.*

Additional limitations include source-selection bias, unverified one-sided narratives, possible residual privacy risk, changing product terminology, distribution shift, and the possibility that users misread calibrated confidence as certainty. The balanced sample does not reflect real prevalence, latency measurements are environment-dependent, and no causal reduction in handling time or error has been demonstrated. Future evaluation should pre-register the analysis, preserve a genuinely untouched holdout, document the serving environment, and involve domain reviewers in defining acceptable errors and escalation rules.

# Reference & Source

## Dataset and development sources

Consumer Financial Protection Bureau. (2026). Consumer Complaint Database. https://www.consumerfinance.gov/data-research/consumer-complaints/

Python Software Foundation. (2026). Python language reference, version 3.11. https://docs.python.org/3/

scikit-learn developers. (2026a). Probability calibration. https://scikit-learn.org/stable/modules/calibration.html

scikit-learn developers. (2026b). Metrics and scoring: Quantifying the quality of predictions. https://scikit-learn.org/stable/modules/model_evaluation.html

Sentence Transformers. (2026). all-MiniLM-L6-v2 model documentation. https://www.sbert.net/

Streamlit. (2026). Streamlit documentation. https://docs.streamlit.io/

## Academic references

Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. *Proceedings of the 21st International Conference on Machine Learning*. https://doi.org/10.1145/1015330.1015432

Jain, P., Tripathi, S., Garg, T., Sadashiv, N., Kundur, N. C., Phadke, M., Gaikwad, M., et al. (2026). An intelligent transformer based framework for bilingual financial complaint classification. Scientific Reports, 16, Article 20594. https://doi.org/10.1038/s41598-026-51771-w

Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In C. Nedellec & C. Rouveirol (Eds.), Machine learning: ECML-98 (pp. 137-142). Springer. https://doi.org/10.1007/BFb0026683

Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 20*(3), 226-239. https://doi.org/10.1109/34.667881

Large, J., Lines, J., & Bagnall, A. (2019). A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates. *Data Mining and Knowledge Discovery, 33*(6), 1674-1709. https://doi.org/10.1007/s10618-019-00638-y

McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. AAAI-98 Workshop on Learning for Text Categorization, 41-48. https://aaai.org/papers/041-ws98-05-007/

Miranda, F. M., Kohnecke, N., & Renard, B. Y. (2023). HiClass: A Python library for local hierarchical classification compatible with scikit-learn. Journal of Machine Learning Research, 24(29), 1-17. https://jmlr.org/papers/v24/21-1518.html

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP 2019, 3982-3992. https://doi.org/10.18653/v1/D19-1410

## Project artifacts consulted

Repository-local plans, manifests, generated metrics, confusion matrices, model metadata, source code, application code, and test sources were consulted as implementation evidence. They are not presented as external scholarship.

**Plagiarism Statement Form (student 1)**

I, Name (Block Capitals) \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_Student ID\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Programme \_\_\_\_\_\_\_\_\_Tutorial Group \_\_\_\_\_\_\_\_\_\_\_\_\_confirm that the submitted work are all my own work and is in my own words.

I \_\_\_\_\_\_\_\_\_\_\_\_\_\_ (Student Name) acknowledge the use of AI generative technology.

Signature : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Date : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Plagiarism Statement Form (student 2)**

I, Name (Block Capitals) \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_Student ID\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Programme \_\_\_\_\_\_\_\_\_Tutorial Group \_\_\_\_\_\_\_\_\_\_\_\_\_confirm that the submitted work are all my own work and is in my own words.

I \_\_\_\_\_\_\_\_\_\_\_\_\_\_ (Student Name) acknowledge the use of AI generative technology.

Signature : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Date : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Plagiarism Statement Form (student 3)**

I, Name (Block Capitals) \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_Student ID\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Programme \_\_\_\_\_\_\_\_\_Tutorial Group \_\_\_\_\_\_\_\_\_\_\_\_\_confirm that the submitted work are all my own work and is in my own words.

I \_\_\_\_\_\_\_\_\_\_\_\_\_\_ (Student Name) acknowledge the use of AI generative technology.

Signature : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Date : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Appendix B**

**Student Free-Rider Report Form**

1.  Name of Student(s) Reported :

2.  Description of Conduct (including time, place, and other relevant details):

3. Supporting Evidence Attached:

☐ Task logs ☐ Meeting minutes ☐ Messages/emails ☐ Drafts/files ☐ Other: \_\_\_\_\_\_\_\_\_

**Declaration:**

I declare that the information provided is true and made in good faith.

Signature:

Student’s Name:

Student ID:

Date:
