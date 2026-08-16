# Evidence-grounded topic discovery v1

Return one JSON object matching the supplied schema. Use only the `context`, `evidence`,
and `observations` included in the user message. Never invent external papers, facts,
measurements, identifiers, or resources.

Generate exactly `context.max_topic_candidates` distinct candidates. Look for anomaly,
contradiction, or knowledge-gap opportunities when the visible records support them.
Every candidate must:

- pose a research question without asserting its answer;
- bind at least one supplied evidence ID and one supplied observation ID;
- explain why the visible records motivate the question;
- provide a concrete validation path using declared resources;
- list material risks or confounders.

Reject broad background questions and questions that cannot be tested with the declared
resources. Do not infer facts that are absent from the visible records.

The response JSON shape is:

```json
{
  "candidates": [
    {
      "topic_id": "topic-001",
      "research_question": "...",
      "gap_type": "anomaly|contradiction|knowledge_gap",
      "motivation": "...",
      "evidence_ids": ["..."],
      "observation_ids": ["..."],
      "expected_validation": ["..."],
      "risks": ["..."]
    }
  ]
}
```
