# One Targeted Hypothesis Revision

Revise the supplied `original_hypothesis` exactly once using only the listed critiques and the
visible case bundle. Return one complete Hypothesis JSON object with no wrapper and no extra
fields. Set `parent_id` to the original hypothesis ID. The application will assign the final new
hypothesis ID.

Address each actionable revision without changing the selected topic. Use only supplied evidence
and observation IDs. Do not add papers, measurements, identifiers, numeric facts, or mechanisms
that are absent from the visible input. Preserve uncertainty instead of filling an evidence gap
with speculation. Output JSON only.
