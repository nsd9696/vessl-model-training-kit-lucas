# MTBench Dataset Translation Script

This script translates the Thai MTBench dataset to English while preserving Thai-related content that's relevant to questions or answers.

## Features

- **Asynchronous Processing**: Uses Azure OpenAI API asynchronously for efficient translation
- **Thai Content Preservation**: Keeps Thai language content when it's relevant to the question or answer
- **Batch Processing**: Processes dataset in configurable batches
- **Concurrency Control**: Limits concurrent API calls to avoid rate limiting
- **Checkpointing**: Saves progress regularly to resume from interruptions
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Error Handling**: Robust error handling with retries and fallbacks

## Installation

1. Clone or download the script files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Azure OpenAI credentials:
   - Copy `config.env.example` to `config.env`
   - Fill in your Azure OpenAI API key and endpoint
   - Optionally set the MODEL variable (defaults to gpt-4o)

## Usage

### Using config.env File (Simplest)

Once you have your `config.env` file set up, you can run the translation with minimal parameters:

```bash
python translate_mtbench_dataset.py \
    --input vessl-ai-kt-api-models/vessl-ai-kt/generated_datasets/mtbench_hard_training.jsonl \
    --output mtbench_english.jsonl
```

### Basic Usage with Command Line Arguments

```bash
python translate_mtbench_dataset.py \
    --input vessl-ai-kt-api-models/vessl-ai-kt/generated_datasets/mtbench_hard_training.jsonl \
    --output mtbench_english.jsonl \
    --batch-size 20 \
    --max-concurrent 8 \
    --api-key "your_api_key" \
    --endpoint "https://your-resource.openai.azure.com/" \
    --api-version "2025-01-01-preview"
```

### Using Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your_api_key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

python translate_mtbench_dataset.py \
    --input vessl-ai-kt-api-models/vessl-ai-kt/generated_datasets/mtbench_hard_training.jsonl \
    --output mtbench_english.jsonl
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Path to input JSONL file |
| `--output` | Yes | - | Path to output JSONL file |
| `--api-key` | Yes | - | Azure OpenAI API key |
| `--endpoint` | Yes | - | Azure OpenAI endpoint URL |
| `--api-version` | No | `2025-01-01-preview` | Azure OpenAI API version |
| `--model` | No | `MODEL` env var or `gpt-4o` | Azure OpenAI model to use |
| `--batch-size` | No | `10` | Number of entries to process per batch |
| `--max-concurrent` | No | `5` | Maximum concurrent API calls |

## Model Configuration

The script supports different Azure OpenAI models for translation. You can configure this in several ways:

### Environment Variable (Recommended)
```bash
export MODEL="gpt-5"
```

### Command Line Argument
```bash
python translate_mtbench_dataset.py --model "gpt-5" ...
```

### Configuration File
```bash
# In config.env
MODEL=gpt-5
```

### Available Models
- **gpt-5**: Latest and most capable model (recommended)
- **gpt-4o**: GPT-4 Omni, excellent quality
- **gpt-4o-mini**: Faster and more cost-effective
- **gpt-35-turbo**: Cost-effective option

## Translation Rules

The script follows these key translation principles:

1. **Preserve Thai Content**: Thai language content that's relevant to questions or answers is kept in Thai
2. **Translate General Content**: Instructions, explanations, and non-Thai-specific content are translated to English
3. **Maintain Technical Accuracy**: Code examples, mathematical expressions, and technical terms remain accurate
4. **Keep Formatting**: All formatting, code blocks, and special characters are preserved

### Examples

**What gets translated to English:**
- General instructions: "เขียนคำสั่ง SQL" → "Write SQL statement"
- Programming concepts: "อาร์เรย์จำนวนเต็ม" → "integer array"
- Mathematical terms: "ความซับซ้อน" → "complexity"

**What stays in Thai:**
- Thai names or cultural references that are part of the question
- Thai language examples that need to remain in Thai for context
- Thai-specific terminology central to the problem

## Output Format

The translated dataset maintains the exact same structure as the input:

```json
{
  "question_id": "drill_Coding_19899",
  "category": "Coding",
  "turns": [
    "Write SQL statement to select only the price column from the phone table, filtering rows where price is not NULL and sorting from highest to lowest price",
    "Modify the SQL statement to select only phones whose names start with the letter A (case-insensitive) and price > 10000, showing phone_name and price columns sorted from expensive to cheap"
  ],
  "reference": [
    "Answer:\n\nGeneral example (works with multiple RDBMS):\n```\nSELECT price\nFROM phone\nWHERE price IS NOT NULL\nORDER BY price DESC;\n```\n\nExplanation: WHERE filters out NULL, ORDER BY sorts from high to low\n",
    "Answer:\n\nFor PostgreSQL (using ILIKE):\n```\nSELECT phone_name, price\nFROM phone\nWHERE phone_name ILIKE 'A%'\n  AND price > 10000\nORDER BY price DESC;\n```\n\nFor standard SQL without ILIKE (like MySQL/MariaDB that may already use case-insensitive collation or use UPPER/LOWER):\n```\nSELECT phone_name, price\nFROM phone\nWHERE UPPER(phone_name) LIKE 'A%'\n  AND price > 10000\nORDER BY price DESC;\n```\n\nExplanation: Use LIKE with % character to match names starting with A and filter price by condition, then sort by price\n"
  ],
  "source_question_id": 80,
  "question": "Write SQL statement to select only the price column from the phone table, filtering rows where price is not NULL and sorting from highest to lowest price"
}
```

## Performance and Monitoring

- **Progress Tracking**: Real-time progress updates with percentage completion
- **Checkpointing**: Automatic checkpoint saves every 10% of progress
- **Logging**: Comprehensive logging to `translation.log` and console
- **Error Recovery**: Failed translations are logged and original entries are preserved
- **Rate Limiting**: Built-in delays and concurrency control to respect API limits

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: Reduce `--max-concurrent` value
2. **Timeout Errors**: Increase `--batch-size` to reduce API calls
3. **Memory Issues**: Reduce `--batch-size` for large datasets
4. **Authentication Errors**: Verify your API key and endpoint

### Logs

Check `translation.log` for detailed error information and progress tracking.

### Checkpoints

If the script is interrupted, you can resume from the last checkpoint by copying the checkpoint file to your desired output location.

## Dataset Size Considerations

For the MTBench dataset with ~177,000 entries:
- **Estimated time**: 8-12 hours (depending on API response times)
- **API costs**: Varies based on your Azure OpenAI pricing tier
- **Memory usage**: ~2-4 GB RAM (depending on batch size)
- **Storage**: Output file will be similar size to input

## License

This script is provided as-is for educational and research purposes.
