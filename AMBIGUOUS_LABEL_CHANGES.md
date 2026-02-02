# Ambiguous Label Implementation Summary

## Overview
Added support for "Ambiguous Label" to the LLM Oracle Labeling system. This label is used in two scenarios:
1. When LLMs explicitly classify a PR as ambiguous (unclear evidence)
2. When models fail to reach consensus threshold (automatic assignment)

## Changes Made

### 1. Configuration (`llm_oracle_labeling/config.py`)
- Added `'Ambiguous Label'` to `RISK_TYPE_LABELS` list
- This makes it a valid label that LLMs can return

### 2. Prompt Builder (`llm_oracle_labeling/prompt_builder.py`)
- Added "Ambiguous Label" definition to the risk_type_labels section:
  - Use when SZZ issues exist but provide insufficient/contradictory evidence
  - Indicates human review needed due to unclear failure descriptions
- Added label selection rules:
  - Use only when evidence is genuinely unclear or contradictory
  - Not a fallback for uncertainty - LLMs should make best judgment

### 3. Response Handler (`llm_oracle_labeling/response_handler.py`)
- Added validation rule: "Ambiguous Label" must appear alone (like "Non-risky")
- If LLM returns Ambiguous Label with other labels, keep only Ambiguous Label
- Ensures consistency in ambiguous case handling

### 4. Consensus Merger (`llm_oracle_labeling/merger.py`)
- **Key Change**: When models fail to reach consensus threshold:
  - Automatically assign `agreed_labels = ('Ambiguous Label',)`
  - Add explanation: "Models did not reach consensus threshold. Human review required."
  - Confidence set to 0.0
- All non-consensus cases now go to human review with Ambiguous Label

### 5. Analysis (`llm_oracle_labeling/analysis.py`)
- No changes needed - already handles all labels dynamically
- Will automatically count and report Ambiguous Label instances

## Behavior

### Scenario 1: LLM Explicitly Returns Ambiguous Label
```json
{
  "risk_type_labels": ["Ambiguous Label"],
  "explanations": [{
    "label": "Ambiguous Label",
    "confidence": 0.5,
    "rationale": "SZZ issues provide contradictory evidence..."
  }]
}
```
- If 2+ models agree on Ambiguous Label → Goes to `accepted_labels.csv`
- If only 1 model returns it → Goes to `human_needed.csv`

### Scenario 2: Models Disagree (No Consensus)
Example: Model A says "Bug Risk", Model B says "Security Risk", Model C says "Performance Risk"
- No 2/3 consensus reached
- System automatically assigns: `agreed_labels = "Ambiguous Label"`
- Goes to `human_needed.csv`
- Explanation: "Models did not reach consensus threshold. Human review required."

## Files Modified
1. ✅ `llm_oracle_labeling/config.py` - Added label to list
2. ✅ `llm_oracle_labeling/prompt_builder.py` - Added to prompt instructions
3. ✅ `llm_oracle_labeling/response_handler.py` - Added validation logic
4. ✅ `llm_oracle_labeling/merger.py` - Auto-assign on no consensus
5. ✅ `uncertainty_selection/config.py` - Fixed unrelated XGBoost error

## Testing Recommendations
1. Run pipeline with small dataset to verify:
   - LLMs can return Ambiguous Label
   - Non-consensus cases get Ambiguous Label assigned
   - human_needed.csv contains Ambiguous Label entries
   
2. Check output CSVs:
   - `accepted_labels_N.csv` - May contain Ambiguous Label if models agreed
   - `human_needed_N.csv` - Will contain Ambiguous Label for disagreements

3. Verify analysis reports Ambiguous Label in label distribution

## Backward Compatibility
- ✅ All existing labels still work
- ✅ Consensus logic unchanged for agreeing models
- ✅ Only adds new behavior for non-consensus cases
- ✅ No breaking changes to data structures or APIs
