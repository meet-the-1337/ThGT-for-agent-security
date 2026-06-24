"""Convert the dataset to Parquet format for HuggingFace compatibility"""

import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

dataset_dir = Path(__file__).parent

def convert_to_parquet():
    """Convert all splits to Parquet format"""
    
    data_dir = dataset_dir / "data"
    
    # Split configurations
    splits = [
        "security_data/without_defense",
        "security_data/with_defense",
        "safety_data/without_defense",
        "safety_data/with_defense"
    ]
    
    for split_path in splits:
        print(f"Processing {split_path}...")
        
        split_dir = data_dir / split_path
        manifest_file = split_dir / "attack_manifest.jsonl"
        trace_dir = split_dir / "traces"
        
        # Load all examples
        examples = []
        with open(manifest_file, 'r') as f:
            for line in f:
                snapshot = json.loads(line)
                trace_id = snapshot.get('result', {}).get('trace_id', '')
                
                # Load corresponding trace file
                trace_file = trace_dir / f"trace_{trace_id}.json"
                
                if not trace_file.exists():
                    print(f"Warning: Trace file not found: {trace_file}")
                    continue
                
                with open(trace_file, 'r') as tf:
                    traces = json.load(tf)
                
                # Clean empty dicts (convert to None) for Parquet compatibility
                def clean_empty_dicts(obj):
                    if isinstance(obj, dict):
                        cleaned = {k: clean_empty_dicts(v) for k, v in obj.items()}
                        # Replace empty dicts with None
                        return {k: (None if (isinstance(v, dict) and not v) else v) 
                                for k, v in cleaned.items()}
                    elif isinstance(obj, list):
                        return [clean_empty_dicts(item) for item in obj]
                    return obj
                
                cleaned_snapshot = clean_empty_dicts(snapshot)
                cleaned_traces = clean_empty_dicts(traces)
                
                # Keep as nested structures (not JSON strings)
                examples.append({
                    "trace_id": trace_id,
                    "attack_snapshot": cleaned_snapshot,  # Dict
                    "trace": cleaned_traces,  # List
                })
        
        # Convert to HuggingFace dataset (will auto-infer nested schema)
        dataset = Dataset.from_list(examples)
        
        # Save as parquet
        parquet_file = split_dir / "data-00000-of-00001.parquet"
        dataset.to_parquet(str(parquet_file))
        
        print(f"  Created {parquet_file} with {len(examples)} examples")

if __name__ == "__main__":
    convert_to_parquet()

