# datasets/aila — Aila identity / personality / company behaviour

**Purpose:** the LAST training stage — Aila's identity, company facts,
response formatting, and persona. Deliberately small and separate so it
does not contaminate general language pretraining (spec objective 5): a
handful of identity facts trained into the pretraining corpus would teach
the model to recite them, not to use language.

The deterministic router and conversational infrastructure (unchanged) sit
on top of the model and already handle most identity/tool behaviour; this
dataset is for the model-level persona in the fine-tune stage.

Existing sources: `datasets/aila_knowledge/aila_company.jsonl`,
`datasets/aila_knowledge/portuguese_basic.jsonl`.
