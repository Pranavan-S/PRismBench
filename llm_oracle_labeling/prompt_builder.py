import json
from typing import Dict
from .config import MAX_PROMPT_CHARS

SYSTEM_INSTRUCTIONS = {
    "task": (
        "Analyze a pull request (PR) and determine its symptom-level risk types "
        "based strictly on SZZ-linked origin issues and observable failure evidence."
    ),
    "expected_output_schema": {
        "risk_type_labels": (
            "array of strings — use ONLY the predefined risk labels. "
            "If non-risky, return exactly ['Non-risky']."
        ),
        "explanations": [
            {
                "label": "string — one of the selected risk type labels",
                "confidence": (
                    "number between 0.0 and 1.0 representing confidence based on evidence strength"
                ),
                "rationale": (
                    "Concise explanation grounded explicitly in SZZ origin issues "
                    "and observable failure symptoms"
                )
            }
        ],
        "IMPORTANT": (
            "Return VALID JSON ONLY. "
            "The 'confidence' field MUST be a numeric value, not a string."
        )
    },

    "core_principle": (
        "You must base ALL risk decisions on empirically observed failures reported "
        "in SZZ origin issues. Do NOT speculate, predict, or infer hypothetical risks."
    ),

    "context_definition": {
        "pull_request": (
            "A pull request (PR) represents a set of code changes that were merged "
            "into the repository."
        ),
        "szz_origin_issues": (
            "SZZ origin issues are issues or bug reports created AFTER the PR, "
            "which were backtraced to this PR using the SZZ algorithm. "
            "They represent observable failures introduced by the PR."
        ),
        "risk_definition": (
            "Risk refers ONLY to real failures observed in production or testing "
            "and reported in SZZ origin issues. Risk does NOT include potential, "
            "theoretical, or hypothetical problems."
        )
    },

    "risk_type_labels": {
        "Bug Risk": (
            "Observable functional or logical failures such as incorrect behavior, "
            "wrong outputs, crashes, exceptions, test failures, deadlocks, race conditions or broken features."
        ),
        "Security Risk": (
            "Observable failures related to vulnerabilities such as authentication flaws, "
            "authorization bypasses, data leaks, privilege escalation, or policy violations."
        ),
        "Performance Risk": (
            "Observable failures involving performance degradation, increased latency, "
            "resource exhaustion, throughput reduction, or scalability regressions."
        ),
        "Maintainability Risk": (
            "Failures caused by design or maintainability problems that later resulted "
            "in regressions, fragile behavior, or difficulty fixing bugs."
        ),
        "Non-risky": (
            "The PR has NO SZZ origin issues and NO observable failures attributed to it."
        ),
        "Ambiguous Label": (
            "Use this ONLY when the SZZ origin issues exist but provide insufficient, "
            "contradictory, or unclear evidence to confidently assign a specific risk type. "
            "This indicates the case needs human review due to ambiguity in the failure description."
        )
    },

    "mandatory_decision_process": [
        "Step 1: Determine whether SZZ origin issues exist.",
        "Step 2: If NO SZZ origin issues exist, return ONLY ['Non-risky'].",
        "Step 3: If SZZ origin issues exist, you MUST assign at least one risk label.",
        "Step 4: For each SZZ origin issue, identify the primary observable failure symptom.",
        "Step 5: Map each observable failure to the MOST DIRECT risk type label."
    ],

    "label_selection_rules": [
        "If SZZ origin issues exist, NEVER return 'Non-risky'.",
        "Select Performance Risk ONLY if the issue explicitly reports latency, throughput, memory, CPU, or scalability degradation.",
        "Multiple labels may be selected ONLY if the failures are independent and observable.",
        "If one risk is a downstream consequence of another, select ONLY the primary symptom-level risk.",
        "Use 'Ambiguous Label' ONLY when you cannot confidently determine the risk type due to unclear or contradictory evidence in the SZZ issues.",
        "Do NOT use 'Ambiguous Label' as a fallback or when you're unsure - make your best judgment based on available evidence."
    ],

    "anti_hallucination_rules": [
        "Do NOT invent failure types not mentioned in SZZ origin issues.",
        "Do NOT infer intent or developer mistakes.",
        "Do NOT use words like 'might', 'could', 'possibly', or 'likely'.",
        "If evidence is weak or vague, explain the uncertainty but still assign the most appropriate label."
    ],

    "evidence_requirement": (
        "Each selected risk label MUST be supported by explicit evidence "
        "from SZZ origin issues. Reference concrete failure descriptions "
        "such as crashes, incorrect output, exceptions, or regressions."
    ),
}


def build_prompt(pr: Dict) -> str:
    szz_issues_raw = pr.get("szz_origin_issues", "[]")
    if isinstance(szz_issues_raw, str):
        try:
            szz_issues = json.loads(szz_issues_raw)
        except Exception:
            szz_issues = []
    elif isinstance(szz_issues_raw, list):
        szz_issues = szz_issues_raw
    else:
        szz_issues = []

    ALLOWED_ISSUE_FIELDS = {
        "issue_key",
        "key",
        "title",
        "summary",
        "description",
        "priority",
    }

    cleaned_issues = []
    for issue in szz_issues:
        if isinstance(issue, dict):
            cleaned_issues.append({
                k: issue[k]
                for k in ALLOWED_ISSUE_FIELDS
                if k in issue
            })

    pr_features = {
        "pr_title": pr.get("pr_title", ""),
        "pr_description": pr.get("pr_description", "")
    }

    kajson_prompt = {
        "szz_origin_issues": cleaned_issues,
        "risky_pull_request_details": pr_features
    }

    prompt_str = json.dumps(kajson_prompt, indent=2)

    if len(prompt_str) > MAX_PROMPT_CHARS:
        kajson_prompt["truncation_notice"] = (
            "Prompt was truncated to fit the LLM context window."
        )
        prompt_str = json.dumps(kajson_prompt, indent=2)

    return prompt_str
