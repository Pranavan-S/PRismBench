# Configuration constants

# Risk type labels
RISK_TYPE_LABELS = [
    'Bug Risk',
    'Security Risk',
    'Performance Risk',
    'Maintainability Risk',
    'Non-risky',
    'Ambiguous Label'
]

# Pipeline settings
MAX_ITEMS = None
FLUSH_EVERY = 50
LLM_SLEEP = 0.5
CONSENSUS_THRESHOLD = 2

# Timeout settings
LLM_TIMEOUT = 30
OLLAMA_TIMEOUT = 180

# Prompt size limits
MAX_PROMPT_CHARS = 80000
MAX_DIFF_CHARS = 10000
MAX_COMMENTS_PER_ISSUE = 0
MAX_COMMENT_CHARS = 10000

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model configuration
MODELS_CONFIG = [
    {'name': 'Gemma', 'model_id': 'gemma2:9b', 'query_fn': 'query_gemma'},
    {'name': 'Llama', 'model_id': 'llama3.1:8b', 'query_fn': 'query_llama'},
    {'name': 'Mistral', 'model_id': 'mistral:7b-instruct', 'query_fn': 'query_mistral'}
]

# Output file configuration
ACCEPTED_LABELS_DIR = 'AcceptedLabels'
HUMAN_REVIEW_DIR = 'HumanEscalation'
MODEL_OUTPUTS_BASE_DIR = 'ModelIntermediateOutputs'

PER_MODEL_FILE_PATTERN = '{model_name}_predictions_{loop_number}.csv'
ACCEPTED_FILE_PATTERN = 'accepted_labels_{loop_number}.csv'
HUMAN_REVIEW_FILE_PATTERN = 'human_needed_{loop_number}.csv'
