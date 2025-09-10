import os, json, time, faiss, numpy as np
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Iterable, Dict, Any
from datasets import load_dataset, Dataset
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import openai
from utils import get_project_root
from tqdm import tqdm

# ──────────────────────────────────────────
# 0. 환경 설정 - Azure OpenAI
# ──────────────────────────────────────────

project_root = get_project_root()

dataset = input("Enter the dataset:\n1. thaiexam\n2. thai_mtbench\n3. viet_mtbench\n")
if dataset == "1":
    dataset = "thaiexam"
elif dataset == "2":
    dataset = "thai_mtbench"
elif dataset == "3":
    dataset = "viet_mtbench"
else:
    print("Invalid dataset. Choose between\n1. thaiexam\n2. thai_mtbench\n3. viet_mtbench.")
    exit()

if 'mtbench' in dataset:
    subject = input("Enter the subject:\n0. All\n1. coding\n2. extraction\n3. knowledge\n4. math\n5. reasoning\n6. roleplay\n7. social_science\n8. stem\n9. writing\n")
    number_to_subject = {0: "all", 1: "coding", 2: "extraction", 3: "knowledge", 4: "math", 5: "reasoning", 6: "roleplay", 7: "social_science", 8: "stem", 9: "writing"}
    if int(subject) in number_to_subject:
        subject = number_to_subject[int(subject)]
    else:
        print("Invalid subject")
        exit()
    if subject == "coding":
        INPUT  = project_root + f"output/{dataset}/drill_data_coding.jsonl"
    elif subject == "extraction":
        INPUT  = project_root + f"output/{dataset}/drill_data_extraction.jsonl"
    elif subject == "knowledge":
        INPUT  = project_root + f"output/{dataset}/drill_data_knowledge.jsonl"
    elif subject == "math":
        INPUT  = project_root + f"output/{dataset}/drill_data_math.jsonl"
    elif subject == "reasoning":
        INPUT  = project_root + f"output/{dataset}/drill_data_reasoning.jsonl"
    elif subject == "roleplay":
        INPUT  = project_root + f"output/{dataset}/drill_data_roleplay.jsonl"
    elif subject == "social_science":
        INPUT  = project_root + f"output/{dataset}/drill_data_social_science.jsonl"
    elif subject == "stem":
        INPUT  = project_root + f"output/{dataset}/drill_data_stem.jsonl"
    elif subject == "writing":
        INPUT  = project_root + f"output/{dataset}/drill_data_writing.jsonl"
    elif subject == "all":
        INPUT  = project_root + f"output/{dataset}/drill_data.jsonl"
    else:
        print("Invalid subject")
        exit()
else:
    INPUT  = project_root + f"output/{dataset}/drill_data.jsonl"
    subject = "all"

if subject == "all":
    OUT_RAPID = project_root + f"output/{dataset}/rapid_dedup.jsonl"
else:
    OUT_RAPID = project_root + f"output/{dataset}/rapid_dedup_{subject}.jsonl"

#OUT_FINAL = "./dedup_drill_data.jsonl"
SIM_THRESHOLD_RAPID = 95          # RapidFuzz 문자 유사도 기준
K_NEIGHBORS = 8                   # 각 문장당 후보 k
COS_THRESHOLD = 0.85              # 임베딩 유사도 cut
LLM_MODEL = "gpt-4o"

# Azure OpenAI 설정
azure_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint="https://kt-aipt-2025-openai.openai.azure.com/"
)

# Azure 모델 매핑
AZURE_DEPLOYMENT_MAP = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4": "gpt4-prod",
    "gpt-4-turbo": "gpt4-turbo-prod"
}

# ──────────────────────────────────────────
# 1. RapidFuzz 문자 중복 제거
# ──────────────────────────────────────────
def fuzz_pair(a: str, pool: List[str], thr: float):
    return [(a, b, s) for b, s, _ in process.extract(a, pool, scorer=fuzz.ratio, score_cutoff=thr)]

def rapid_dedup(ds: Dataset, column="question", thr=95) -> Dataset:
    vals = [str(x) for x in ds[column]]
    uniq = list(set(vals))
    mpair = partial(fuzz_pair, pool=uniq, thr=thr)
    with Pool(cpu_count()) as P:
        matches = list(tqdm(P.imap(mpair, uniq, 100),
                            total=len(uniq), desc="RapidFuzz"))
    str2idx = defaultdict(list)
    for i, v in enumerate(vals): str2idx[v].append(i)
    remove = set()
    for lst in matches:
        for a,b,_ in lst:
            dup = sorted(set(str2idx[a]+str2idx[b]))
            remove.update(dup[1:])            # 첫 행만 남김
    return ds.select([i for i in range(len(ds)) if i not in remove])


# ──────────────────────────────────────────
# Main execution with enhanced pipeline
# ──────────────────────────────────────────
def main():
    print("🚀 Starting Enhanced MT-Bench Thai Deduplication Pipeline")
    print(f"📁 Input: {INPUT}")
    print(f"🔧 Using Azure OpenAI: {LLM_MODEL} -> {AZURE_DEPLOYMENT_MAP.get(LLM_MODEL, LLM_MODEL)}")
    
    # 0) Load dataset
    print("\n📖 Loading dataset...")
    ds = load_dataset("json", data_files=INPUT, split="train")
    print(f"Original dataset size: {len(ds)}")
    
    # Add question column for processing
    ds = ds.map(lambda ex: {"question": ex["turns"][0] if ex.get("turns") else ""}, desc="Extract questions")

    # 1) RapidFuzz character-based deduplication
    print("\n⚡ Step 1: RapidFuzz character-based deduplication...")
    ds1 = rapid_dedup(ds, "question", SIM_THRESHOLD_RAPID)
    ds1.to_json(OUT_RAPID, orient="records", lines=True, force_ascii=False)
    print(f"After RapidFuzz: {len(ds1)} (removed {len(ds) - len(ds1)})")
 
    print(f"\n✅ Deduplication complete!")
    print(f"📊 Summary:")
    print(f"   Original: {len(ds):,}")
    print(f"   After RapidFuzz: {len(ds1):,}")

if __name__ == "__main__":
    main()