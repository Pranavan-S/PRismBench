import json
from typing import Dict, List, Tuple
from .config import RISK_TYPE_LABELS


def parse_llm_json(response_text: str) -> Dict:
    try:
        return json.loads(response_text)
    except Exception as e1:
        text = response_text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                json_str = text[start:end+1]
                return json.loads(json_str)
            except Exception as e3:
                print(f'Could not parse JSON.')
                print(f'Response length: {len(response_text)} chars')
                print(f'First 800 chars: {response_text[:800]}')
                print(f'Last 200 chars: {response_text[-200:]}')
                print(f'Parse errors: {str(e1)[:100]}, {str(e3)[:100]}')

    raise ValueError('Could not parse JSON from LLM response')


def normalize_labels(labels: List[str]) -> Tuple[str]:
    if not labels:
        return tuple()

    canonical = []
    for lbl in labels:
        lbl_norm = lbl.strip()
        found = next((l for l in RISK_TYPE_LABELS if l.lower() == lbl_norm.lower()), None)
        if found:
            canonical.append(found)
        else:
            canonical.append(lbl_norm)

    return tuple(sorted(set(canonical)))


def validate_llm_response(response: Dict, model_name: str, pr_number: int) -> Dict:
    errors = []
    if 'risk_type_labels' not in response and 'labels' not in response:
        errors.append("Missing 'risk_type_labels' or 'labels' key")

    if 'explanations' not in response:
        errors.append("Missing 'explanations' key")
    if 'labels' in response and 'risk_type_labels' not in response:
        response['risk_type_labels'] = response['labels']
    if 'risk_type_labels' not in response:
        response['risk_type_labels'] = []

    if 'explanations' not in response:
        response['explanations'] = []
    labels = response['risk_type_labels']
    if not isinstance(labels, list):
        errors.append(f"risk_type_labels should be a list, got {type(labels)}")
        response['risk_type_labels'] = []
        labels = []
    unknown_labels = [l for l in labels if l not in RISK_TYPE_LABELS]
    if unknown_labels:
        print(f'WARNING: {model_name} for PR {pr_number} returned unknown labels: {unknown_labels}')
        response['risk_type_labels'] = [l for l in labels if l in RISK_TYPE_LABELS]
    if 'Non-risky' in response['risk_type_labels']:
        if len(response['risk_type_labels']) > 1:
            print(f'WARNING: {model_name} for PR {pr_number} returned Non-risky with other labels, keeping only Non-risky')
            response['risk_type_labels'] = ['Non-risky']
            response['explanations'] = [e for e in response['explanations'] if e.get('label') == 'Non-risky']
    
    # Handle Ambiguous Label - it should be alone
    if 'Ambiguous Label' in response['risk_type_labels']:
        if len(response['risk_type_labels']) > 1:
            print(f'WARNING: {model_name} for PR {pr_number} returned Ambiguous Label with other labels, keeping only Ambiguous Label')
            response['risk_type_labels'] = ['Ambiguous Label']
            response['explanations'] = [e for e in response['explanations'] if e.get('label') == 'Ambiguous Label']
    explanations = response['explanations']
    if not isinstance(explanations, list):
        print(f'WARNING: {model_name} explanations not a list, converting')
        response['explanations'] = []

    if errors:
        print(f'ERROR: {model_name} validation errors for PR {pr_number}: {errors}')

    return response
