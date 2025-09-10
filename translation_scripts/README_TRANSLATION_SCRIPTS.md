# Translation Scripts Directory

This directory contains all files related to translating Thai datasets (MTBench and ThaiExam) to English while preserving Thai-related content that's relevant to questions or answers.

## 📁 File Organization

### Main Translation Scripts
- `translate_mtbench_dataset.py` - Main translation script for MTBench dataset
- `translate_thaiexam_dataset.py` - Main translation script for ThaiExam dataset

### Test Scripts
- `test_translation.py` - Test script for MTBench translation
- `test_thaiexam_translation.py` - Test script for ThaiExam translation

### Shell Scripts
- `run_translation.sh` - Shell script to run MTBench translation
- `run_thaiexam_translation.sh` - Shell script to run ThaiExam translation

### Configuration Files
- `setup_config.py` - Interactive setup script for configuration
- `config.env` - Environment configuration file (supports multi-endpoint configuration)
- `config.env.example` - Example configuration file
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Main documentation for MTBench translation
- `README_thaiexam.md` - Documentation for ThaiExam translation
- `SAFETY_IMPROVEMENTS.md` - Safety improvements documentation

### Log Files
- `translation.log` - General translation log file
- `thaiexam_translation.log` - ThaiExam-specific translation log
- `generation.log` - Generation log file

### Data Files
- `mtbench_drills_total_1000_eng_gpt_translated.json` - Translated MTBench dataset

## 🚀 Quick Start

1. **Setup Configuration:**
   ```bash
   python setup_config.py
   ```

2. **Run MTBench Translation:**
   ```bash
   ./run_translation.sh
   ```

3. **Run ThaiExam Translation:**
   ```bash
   ./run_thaiexam_translation.sh
   ```

## 📋 Features

- **Preserves Thai Content**: Keeps Thai language content relevant to questions/answers
- **Batch Processing**: Handles large datasets efficiently
- **Concurrency Control**: Manages API rate limits
- **Checkpointing**: Resume interrupted translations
- **Progress Tracking**: Real-time progress monitoring
- **Comprehensive Logging**: Detailed logs for debugging

## 🔧 Requirements

- Python 3.8+
- Azure OpenAI API access
- Dependencies listed in `requirements.txt`

## 📝 Notes

- All scripts use Azure OpenAI API for translation
- Thai content relevant to questions/answers is preserved in Thai
- General instructions and explanations are translated to English
- Supports both MTBench and ThaiExam dataset formats
- **Content Filter Protection**: Scripts include automatic content sanitization to avoid Azure OpenAI content filter violations
- **Enhanced Error Handling**: Specific handling for content filter, rate limiting, and API errors
- **Fallback Strategy**: Original text is preserved when translation fails

## 🌐 Multi-Endpoint Configuration

The translation scripts now support **automatic endpoint selection** based on the model you choose:

### 📡 **Endpoint Mapping:**
- **GPT-5 models** → Use `GPT5_*` configuration
- **GPT-4.1, GPT-4o, etc.** → Use `GPT4_*` configuration  
- **Fallback** → Legacy `AZURE_OPENAI_*` variables

### ⚙️ **Configuration Format:**
```bash
# config.env
# GPT-5 Endpoint
GPT5_API_KEY=your_gpt5_api_key
GPT5_ENDPOINT=https://your-gpt5-resource.cognitiveservices.azure.com/openai/v1/
GPT5_DEPLOYMENT_NAME=gpt-5
GPT5_API_VERSION=2025-01-01-preview

# GPT-4 Endpoint  
GPT4_API_KEY=your_gpt4_api_key
GPT4_ENDPOINT=https://your-gpt4-resource.cognitiveservices.azure.com/openai/v1/
GPT4_DEPLOYMENT_NAME=gpt-4.1
GPT4_API_VERSION=2025-01-01-preview

# Legacy (fallback)
AZURE_OPENAI_API_KEY=fallback_key
AZURE_OPENAI_ENDPOINT=fallback_endpoint
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

### 🎯 **Usage:**
- **No code changes needed** - Just specify your desired model
- **Automatic selection** - Scripts choose the right endpoint
- **Backward compatible** - Existing configurations still work
- **API Version**: GPT-4.1 requires `2025-01-01-preview` API version (automatically handled)

## 🚨 Recent Improvements Applied

### Content Filter Handling (Improved)
- **Issue**: 684+ content filter violations preventing translations
- **Solution**: Enhanced academic prompts and skip/save system for problematic entries
- **Status**: ✅ Improved - maintains dataset integrity while handling violations

### Academic Translation Prompts (Enhanced)
- **Issue**: Generic prompts triggering content filters
- **Solution**: Academic research-focused prompts emphasizing educational context
- **Status**: ✅ Enhanced - better bypass rate for educational content

### Skipped Entry Management (New)
- **Feature**: Automatic tracking and saving of content filter violations
- **Solution**: Problematic entries are skipped and saved separately for later processing
- **Status**: ✅ Implemented - preserves all original data

### Dataset Integrity (Preserved)
- **Goal**: Keep translations as close to original Thai as possible
- **Solution**: No content sanitization, academic context emphasis, separate handling
- **Status**: ✅ Achieved - original content preserved, problematic entries handled separately

## 🔧 Troubleshooting

### Content Filter Issues
If you encounter content filter violations:
1. **Automatic Handling**: Scripts now automatically skip problematic entries and save them separately
2. **Check Skipped Entries**: Review `.skipped.jsonl` and `.skipped_summary.md` files 
3. **Academic Context**: Enhanced prompts emphasize educational research context
4. **Original Data Preserved**: All original content is maintained - nothing is lost or modified

### API Errors
- **429 Rate Limit**: Script automatically implements exponential backoff
- **404 Resource Not Found**: Check your endpoint and deployment name
- **Missing Arguments**: Verify your config.env file is properly set up

### Testing Improvements
Run the test script to verify the improved system:
```bash
python test_improved_translation.py
```

### Reviewing Skipped Entries
After translation, check for skipped entries:
```bash
# Check if any entries were skipped
ls *.skipped.jsonl *.skipped_summary.md

# Review skipped entries summary
cat your_output_file.skipped_summary.md

# Process skipped entries manually if needed
python process_skipped_entries.py your_output_file.skipped.jsonl
```
