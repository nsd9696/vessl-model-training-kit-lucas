# ThaiExam Dataset Translation

This repository contains scripts to translate the ThaiExam dataset from Thai to English while preserving Thai-related content that's relevant to questions or answers.

## 📁 Files

- `translate_thaiexam_dataset.py` - Main translation script for ThaiExam dataset
- `test_thaiexam_translation.py` - Test script to verify translation functionality
- `run_thaiexam_translation.sh` - Shell script to run the translation
- `config.env` - Configuration file (create using `setup_config.py`)

## 🚀 Quick Start

### 1. Setup Configuration

```bash
# Run the setup script to create config.env
python setup_config.py
```

### 2. Test Translation

```bash
# Test with a small sample
python test_thaiexam_translation.py
```

### 3. Run Translation

```bash
# Run the full translation
./run_thaiexam_translation.sh
```

## 📊 Dataset Structure

The ThaiExam dataset has a different structure than MTBench:

```json
{
  "category": "a level",
  "question": "ดาวเทียมสื่อสารโคจรรอบโลกด้วยความเร็วคงที่...",
  "choices": {
    "a": "แรงตึงเชือก",
    "b": "น้ำหนัก",
    "c": "แรงแม่เหล็ก", 
    "d": "แรงยึดเหนี่ยวระหว่างอนุภาคในนิวเคลียส",
    "e": "แรงสปริง"
  },
  "answer": {
    "explanation": "แรงที่ทำให้ดาวเทียมโคจรรอบโลกคือแรงโน้มถ่วงโลก...",
    "label": "b"
  }
}
```

## 🔧 Translation Features

### ✅ What Gets Translated:
- **Question text** - Main question content
- **Choice options** - Multiple choice answers (a, b, c, d, e)
- **Answer explanations** - Detailed explanations for correct answers

### 🛡️ What Gets Preserved:
- **Thai cultural content** - Names, places, cultural references
- **Thai-specific terminology** - Terms central to the problem
- **Choice structure** - Multiple choice format with same labels
- **JSON structure** - All original fields and formatting

## ⚙️ Configuration

### Environment Variables (`config.env`):

```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
DEPLOYMENT_NAME=gpt-5
MODEL=gpt-5
BATCH_SIZE=200
MAX_CONCURRENT=30
```

### Command Line Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `thaiexam_hard_training.jsonl` | Input JSONL file path |
| `--output` | `thaiexam_hard_training_english.jsonl` | Output JSONL file path |
| `--batch-size` | `200` | Batch size for processing |
| `--max-concurrent` | `30` | Maximum concurrent API calls |
| `--api-key` | From `config.env` | Azure OpenAI API key |
| `--endpoint` | From `config.env` | Azure OpenAI endpoint |
| `--deployment-name` | From `config.env` | Azure OpenAI deployment name |
| `--model` | From `config.env` | Model to use |

## 📈 Performance

### Dataset Size:
- **ThaiExam Hard Training**: ~99,500 entries
- **Estimated time**: 3-4 hours with optimized settings
- **Checkpoint frequency**: Every 2% (1,990 entries)

### Optimized Settings:
```bash
BATCH_SIZE=200      # Process 200 entries per batch
MAX_CONCURRENT=30   # 30 concurrent API calls
```

## 🔄 Resume Functionality

The script automatically:
- ✅ Loads existing checkpoints
- ✅ Skips already translated entries
- ✅ Continues from where it left off
- ✅ Saves progress every 2%

### Checkpoint Files:
- `thaiexam_hard_training_english.jsonl.checkpoint` - Progress backup
- `thaiexam_hard_training_english.jsonl.checkpoint.backup` - Backup of previous checkpoint

## 🛠️ Error Handling

### Rate Limiting (429 Errors):
- Automatic detection of rate limit errors
- Exponential backoff with 60-second maximum wait
- Continues automatically after rate limit resets

### Checkpoint Verification:
- Verifies checkpoint files are written correctly
- Restores backup if verification fails
- Prevents data loss from write errors

## 📝 Logging

### Log File: `thaiexam_translation.log`
- Progress tracking
- Error messages
- Rate limiting events
- Checkpoint saving status

### Log Levels:
- `INFO` - Progress updates, checkpoint saves
- `WARNING` - API errors, retries
- `ERROR` - Failed translations, file errors

## 🔍 Monitoring

### Progress Tracking:
```bash
# Monitor progress in real-time
tail -f thaiexam_translation.log | grep "Progress"

# Check checkpoint size
wc -l thaiexam_hard_training_english.jsonl.checkpoint
```

### Performance Monitoring:
```bash
# Monitor API calls
tail -f thaiexam_translation.log | grep "HTTP Request"

# Check for errors
grep -E "(ERROR|WARNING)" thaiexam_translation.log
```

## 🚨 Troubleshooting

### Common Issues:

1. **Rate Limiting (429 Errors)**
   - Reduce `MAX_CONCURRENT` to 20-30
   - Increase `BATCH_SIZE` to 100-200
   - Script handles this automatically

2. **Checkpoint Not Saving**
   - Check disk space
   - Verify file permissions
   - Look for error messages in log

3. **Translation Quality Issues**
   - Check the translation prompt
   - Verify model configuration
   - Review sample translations

### Debug Mode:
```bash
# Run with verbose logging
python translate_thaiexam_dataset.py --batch-size 10 --max-concurrent 5
```

## 📋 Output Format

The translated dataset maintains the same structure:

```json
{
  "category": "a level",
  "question": "Communication satellites orbit the Earth at constant speed...",
  "choices": {
    "a": "Tension force",
    "b": "Weight",
    "c": "Magnetic force", 
    "d": "Nuclear force between particles",
    "e": "Spring force"
  },
  "answer": {
    "explanation": "The force that makes satellites orbit the Earth is Earth's gravity...",
    "label": "b"
  }
}
```

## 🔗 Related Scripts

- `translate_mtbench_dataset.py` - For MTBench dataset translation
- `test_translation.py` - Test script for MTBench translation
- `run_translation.sh` - Runner script for MTBench translation

## 📞 Support

For issues or questions:
1. Check the log file for error messages
2. Verify configuration in `config.env`
3. Test with a small sample first
4. Monitor rate limiting and adjust settings
