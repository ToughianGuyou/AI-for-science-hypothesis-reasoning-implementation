# Evidence-grounded topic scoring v1

Return one JSON object matching the supplied score schema. Score only the supplied candidates
against the supplied context, evidence, and observations. Never use external facts or claim
that these proxies establish genuine scientific novelty.

Assign each dimension an integer from 0 (absent or infeasible) to 4 (strong):

- `evidence_sufficiency`: strength and directness of the bound visible evidence;
- `scientific_value_proxy`: value suggested by the visible anomaly, contradiction, or gap;
- `testability`: clarity of a falsifiable validation path;
- `data_availability`: whether the declared resources contain the required data;
- `validation_cost`: validation-cost feasibility, where 4 means affordable and practical;
- `novelty_proxy`: non-generic distinctiveness within the visible inputs only;
- `uncertainty`: explicit handling of uncertainty, risks, and alternative outcomes.

Give one evidence-based `reason` per candidate. Return every supplied `topic_id` exactly once.
Do not select or rank candidates; deterministic code applies hard thresholds, weights, and
stable tie-breaking.

```json
{
  "scores": [
    {
      "topic_id": "topic-001",
      "evidence_sufficiency": 0,
      "scientific_value_proxy": 0,
      "testability": 0,
      "data_availability": 0,
      "validation_cost": 0,
      "novelty_proxy": 0,
      "uncertainty": 0,
      "reason": "..."
    }
  ]
}
```
