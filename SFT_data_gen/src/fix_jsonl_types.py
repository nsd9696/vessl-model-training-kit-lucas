#!/usr/bin/env python3
"""
Script to fix data type inconsistency in JSONL file
"""

import json
import os
from pathlib import Path
from utils import (Config, load_config, get_project_root)

def fix_jsonl_types(input_file, output_file, subject):
    """
    Fix data type inconsistency by converting all source_question_id to strings
    """
    print(f"Fixing data types in {input_file}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        input_data = [json.loads(line) for line in infile]

        for i in range(len(input_data) - 1, -1, -1):
            if subject not in ['writing', 'roleplay', 'humanities']:
                if len(input_data[i]['reference']) != 2:
                    del input_data[i]
                    continue
            elif subject in ['writing', 'roleplay', 'humanities']:
                if len(input_data[i]['reference']) != 3:
                    del input_data[i]
                    continue
            for k, reference in enumerate(input_data[i]['reference']):
                if subject == 'writing' or subject == 'roleplay' or subject == 'humanities':
                    for l, example in enumerate(reference):
                        if type(example) == None:
                            del input_data[i]
                            continue
                        elif type(example) == list:
                            fixed_reference = ""
                            for j, item in enumerate(example):
                                fixed_reference += f"- {item}\n"
                            input_data[i]['reference'][k][l] = fixed_reference
                        else:
                            continue
                    
                else:
                    if type(reference) == None:
                        del input_data[i]
                        continue
                    elif type(reference) == list:
                        fixed_reference = ""
                        for j,item in enumerate(reference):
                            fixed_reference += f"{j + 1}. {item}\n"
                        input_data[i]['reference'][k] = fixed_reference
                    else:
                        continue
        
            if 'source_question_id' in input_data[i]:
                original_value = input_data[i]['source_question_id']
                original_type = type(original_value).__name__
                
                # Convert to integer
                try:
                    input_data[i]['source_question_id'] = int(original_value)
                except:
                    if 'unknown' in input_data[i]['source_question_id']:
                        input_data[i]['source_question_id'] = 0
                    else:
                        raise ValueError(f"Invalid source_question_id: {original_value}")
                if original_type != 'int':
                    fixed_count += 1
                    #print(f"Line {line_num}: {original_value} ({original_type}) -> {data['source_question_id']} (str)")
            
            if 'question' not in input_data[i]:
                input_data[i]['question'] = input_data[i]['turns'][0]
    
            
            # Write fixed line
            outfile.write(json.dumps(input_data[i], ensure_ascii=False) + '\n')
            
    print(f"\nFixed {fixed_count} out of {total_count} lines")
    print(f"Output saved to: {output_file}")

def main():
    project_root = get_project_root()
    subject = ["coding", "extraction", "knowledge", "math", "reasoning", "roleplay", "social_science", "stem", "writing", "humanities"]
    skip = ["coding", "extraction", "knowledge", "math","reasoning", "roleplay", "social_science", "stem", "humanities"]
    for subject in subject:
        if subject in skip:
            continue
        input_file = os.path.join(project_root, "output", f"drill_data_{subject}.jsonl")
        output_file = os.path.join(project_root, "output", f"drill_data_{subject}_fixed.jsonl")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} not found. Disregard if the dataset does not have the subject in the first place. Skipping...")
            continue
        
        fix_jsonl_types(input_file, output_file, subject)
        
        # Verify the fix
        print("\n" + "=" * 60)
        print("Verifying fix...")
        
        try:
            from datasets import load_dataset
            ds = load_dataset("json", data_files=output_file, split="train")
            print(f"✅ Successfully loaded fixed dataset: {len(ds)} examples")
        except Exception as e:
            print(f"❌ Error loading fixed dataset: {e}")

if __name__ == "__main__":
    main() 