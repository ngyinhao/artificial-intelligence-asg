# ADR 0001: Select NLP Complaint Routing

- **Status:** Accepted
- **Decision owner:** Single project contributor

## Context

The project requires an AI problem, comparative methods, measurable evaluation, and a
working prototype. The selected direction must be feasible on a local computer and
provide enough methodological depth for critical analysis.

## Decision

Use natural language processing to classify public CFPB consumer complaint narratives
into six financial-product categories. Compare a Naive Bayes baseline, a calibrated
linear SVM, and transformer sentence embeddings with Logistic Regression under one
fixed dataset and evaluation protocol.

Use the official CFPB API rather than web crawling or collecting private complaint
text. Deliver a local Streamlit application that provides a suggested route, calibrated
confidence, top-three candidate categories, comparative metrics, and limitations.

## Decision drivers

- Multiclass routing is more operationally meaningful than generic sentiment analysis.
- The official dataset provides narrative text, product labels, field documentation,
  public access, and de-identification controls.
- The three approaches represent a baseline, a strong sparse linear method, and a
  transformer-based semantic method.
- Macro-F1 and per-class metrics expose uneven routing quality better than accuracy
  alone.
- A local, artifact-only interface avoids operational service dependencies.

## Alternatives rejected

- **Review sentiment:** simpler but common and less useful for support routing.
- **Spam detection:** clear binary evaluation but insufficiently rich for the intended
  comparison.
- **Live web crawling:** unnecessary legal, privacy, labelling, and reproducibility risk.
- **Generative chatbot:** subjective evaluation and hallucination risk do not match the
  classification objective.
- **Full transformer fine-tuning:** higher compute and failure risk for a single local
  contributor; frozen MiniLM embeddings retain semantic representation at lower cost.

## Consequences

- The findings apply only to the sampled, opt-in CFPB narratives and selected labels.
- Product names inside narratives can make routing easier; error analysis must discuss
  how explicit keywords affect performance.
- Published CFPB narratives are scrubbed but still treated as sensitive text and kept
  out of version control.
- The prototype supports human routing and must not assess truthfulness, harm, company
  behavior, or regulatory outcomes.
