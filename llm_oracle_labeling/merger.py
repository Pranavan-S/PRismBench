import json
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm

from .config import CONSENSUS_THRESHOLD
from .consensus import responses_agree


def merge_model_results_and_apply_consensus(
    model_results: Dict[str, pd.DataFrame],
    consensus_threshold: int = CONSENSUS_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print('\n' + '='*80)
    print(' Merging results and applying consensus voting...')
    print('='*80)
    all_pr_numbers = set()
    for df in model_results.values():
        all_pr_numbers.update(df['pr_number'].unique())

    accepted_rows = []
    human_rows = []
    for pr_number in tqdm(sorted(all_pr_numbers), desc='Applying consensus'):
        responses = []
        for model_name, df in model_results.items():
            pr_result = df[df['pr_number'] == pr_number]
            if len(pr_result) == 0:
                continue
            pr_result = pr_result.iloc[0]
            labels = pr_result['risk_type_labels']
            if isinstance(labels, str):
                try:
                    labels = json.loads(labels)
                except:
                    labels = []
            elif not isinstance(labels, list):
                labels = []
            explanations = pr_result['explanations']
            if isinstance(explanations, str):
                try:
                    explanations = json.loads(explanations)
                except:
                    explanations = []
            elif not isinstance(explanations, list):
                explanations = []

            response = {
                'model': model_name,
                'risk_type_labels': labels,
                'explanations': explanations
            }

            if not pr_result['success']:
                response['error'] = pr_result['error']

            responses.append(response)
        agree, agreed_labels, winning_responses = responses_agree(
            responses, 
            consensus_threshold
        )
        
        # If no consensus reached, assign "Ambiguous Label" for human review
        if not agree:
            agreed_labels = ('Ambiguous Label',)
        
        out_row = {
            'pr_number': pr_number,
            'accepted': agree,
            'vote_count': len(winning_responses) if agree else 0,
            'agreed_labels': ';'.join(agreed_labels) if agreed_labels else '',
            'model_responses_summary': json.dumps([
                {
                    'model': r['model'],
                    'labels': r.get('risk_type_labels', []),
                    'error': r.get('error', '')
                }
                for r in responses
            ])
        }
        if agree and agreed_labels and winning_responses:
            first_winner = winning_responses[0]
            explanations_list = first_winner.get('explanations', [])
            explanations_dict = {}
            for exp in explanations_list:
                lbl = exp.get('label', '')
                if lbl in agreed_labels:
                    explanations_dict[lbl] = {
                        'confidence': exp.get('confidence', 0.0),
                        'rationale': exp.get('rationale', '')
                    }
            out_row['explanations'] = json.dumps(explanations_dict)
        elif not agree:
            # For ambiguous cases, provide explanation about the disagreement
            out_row['explanations'] = json.dumps({
                'Ambiguous Label': {
                    'confidence': 0.0,
                    'rationale': 'Models did not reach consensus threshold. Human review required.'
                }
            })
        else:
            out_row['explanations'] = '{}'
        if agree:
            accepted_rows.append(out_row)
        else:
            human_rows.append(out_row)
    accepted_df = pd.DataFrame(accepted_rows)
    human_df = pd.DataFrame(human_rows)
    total = len(accepted_df) + len(human_df)
    print(f' Consensus complete:')
    if total > 0:
        print(f'   Accepted: {len(accepted_df)} ({len(accepted_df)/total*100:.1f}%)')
        print(f'   Need human review: {len(human_df)} ({len(human_df)/total*100:.1f}%)')
    else:
        print('   No PRs processed during merging')

    return accepted_df, human_df
