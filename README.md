---
license: apache-2.0
library_name: transformers
tags:
- laya
- multilingual
- mmbert
- system-one
- calibrated-decisions
- rlcd
- classification
- routing
- guardrails
- moderation
- email-triage
- commercial-use
---

# Laya Multilingual (mmBERT-base)

Laya Multilingual is a fast, non-autoregressive System 1 decision engine supporting over 100 languages natively. It evaluates typed schemas (`choice`, `score`, `noul`) over unstructured state (text, email, ticket, or JSON) in a **single ~35ms forward pass** on a GPU, returning mathematically calibrated probabilities and confidence scores with zero text generation and zero hallucinations.

Trained on over 1,000,000 multi-domain records using **RLCD (Reinforcement Learning for Calibrated Decisions)** with strictly proper scoring rules.

| Question Type | Returns | Real-World Application |
|---|---|---|
| `choice` | Top option, probability distribution, confidence | Multilingual intent classification, department routing, topic sorting |
| `score` | Position along an ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| `noul` | Calibrated boolean probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

## Model Architecture
* **Backbone Encoder:** `jhu-clsp/mmBERT-base` (135M parameters, bidirectional ModernBERT architecture, 22 layers, 768 hidden dimension, 12 attention heads, 256k multilingual vocabulary covering 100+ languages, natively supporting up to 8,192 tokens with RoPE).
* **Decision Head:** 2-layer TransformerEncoder (768 dim, 12 heads, FFN=3072) + Option Marker Scorer + Cost-Sensitive Act/Escalate Head.
* **Total Parameters:** ~160M parameters (runs 3x faster than 400M+ models).
* **Option Markers:** Options are denoted by `[MASK]` tokens; hidden states are pooled via `torch.gather` and scored in a single forward pass.

## Training Details
* **Method:** RLCD (pure policy gradient REINFORCE with GRPO-style group baseline).
* **Rewards:** Strictly proper scoring rules (Log-score + Spherical score + Ranked Probability Score for ordinal rubrics).
* **Dataset:** 100% human-annotated real-world datasets across multilingual intents, NLI, safety moderation, prompt guardrails, and all 5 email triage and threat sources (Enron spam, phishing emails, phishing URLs, support tickets, and Bitext intents).
* **Precision:** Trained in BF16 on NVIDIA RTX 6000 Pro.
* **Fitted Temperatures:** [1.0, 1.0, 1.0]

## Quickstart (`pip install laya`)

```bash
pip install laya
```

```python
import laya

# Load the multilingual model directly from Hugging Face
agent = laya.load("convaiinnovations/laya-multilingual")

# 1. State in any supported language
state = {
    "subject": "Facture débitée deux fois",
    "body": "Bonjour, nous avons été facturés deux fois pour le mois de mars. Veuillez rembourser le doublon aujourd'hui."
}

# 2. Define typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Quel service doit traiter cet e-mail ?",
        "criteria": {
            "facturation": "factures, paiements, remboursements",
            "technique": "bugs, pannes, intégrations",
            "commercial": "tarification, nouveaux contrats",
            "autre": "toute autre demande"
        }
    },
    "urgence": {
        "type": "score",
        "instructions": "Quelle est l'urgence de cette demande ?",
        "criteria": ["pas urgent", "bientôt", "blocage critique ou date limite"]
    },
    "est_spam": {
        "type": "noul",
        "instructions": "Cet e-mail est-il un spam non sollicité ?"
    }
}

# 3. Single forward pass in ~35 ms
result = agent.predict(state, questions)
print(result["answers"])
```

## Interactive Demo
Try the live Space demo: [convaiinnovations/laya-demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)

## License
Released under the Apache 2.0 License by [Convai Innovations](https://huggingface.co/convaiinnovations).
