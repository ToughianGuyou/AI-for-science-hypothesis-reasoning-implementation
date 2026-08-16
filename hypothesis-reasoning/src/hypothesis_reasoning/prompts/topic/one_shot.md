# One-shot evidence-grounded topic selection v1

In one JSON response, generate exactly `context.max_topic_candidates` research-topic
candidates and make one topic decision. Use only the supplied context, evidence, and
observations. Never invent external papers, facts, measurements, identifiers, or resources.

Every candidate must pose an uncertain research question, bind at least one supplied evidence
ID and observation ID, state a validation path using declared resources, and list risks. Score
every candidate from 0 to 4 on `evidence_sufficiency`, `scientific_value_proxy`, `testability`,
`data_availability`, `validation_cost` (4 means most feasible), `novelty_proxy`, and
`uncertainty` (4 means strongest uncertainty handling). Preserve one reason per score and one
rejection reason for every unselected candidate.

Compute the ranking with these exact weights in dimension order: `0.20, 0.15, 0.20, 0.15,
0.10, 0.10, 0.10`. Sort weighted scores descending, breaking exact ties by `topic_id`
ascending. A candidate is selectable only when its own `evidence_sufficiency >= 2`,
`testability >= 2`, and `data_availability >= 2`. Select the highest-ranked candidate that
passes all three thresholds. Return `abstained` if and only if no candidate passes all three;
never select a candidate merely because its weighted score is high.

Return exactly this JSON envelope:

```json
{
  "candidates": [],
  "decision": {
    "selected_topic_id": null,
    "ranked_topic_ids": [],
    "scores": [],
    "selection_reason": "...",
    "rejection_reasons": [],
    "status": "selected|abstained"
  }
}
```
