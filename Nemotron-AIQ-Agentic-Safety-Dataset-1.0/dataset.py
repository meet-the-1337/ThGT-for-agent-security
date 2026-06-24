"""Nemotron-AIQ Agentic Safety Dataset"""

import json
import datasets
from pathlib import Path

_DESCRIPTION = """
Nemotron-AIQ-Agentic-Safety-Dataset is a comprehensive dataset that captures a broad range of novel safety and security contextual risks that can emerge within agentic systems.
"""

_HOMEPAGE = "https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0"

_LICENSE = "NVIDIA Evaluation Dataset License Agreement"

_CITATION = """
@dataset{nemotron_aiq_agentic_safety_2025,
  title={Nemotron-AIQ Agentic Safety Dataset},
  author={Shaona Ghosh and Soumili Nandi and Dan Zhao and Kyriacos Shiarlis and Matthew Fiedler},
  year={2025},
  publisher={Hugging Face},
  note={NVIDIA Corporation},
  url={https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0}
}
"""

class NemotronAIQAgenticSafetyDataset(datasets.GeneratorBasedBuilder):
    """Nemotron-AIQ Agentic Safety Dataset"""

    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="safety",
            version=VERSION,
            description="Safety evaluation data with content safety attacks"
        ),
        datasets.BuilderConfig(
            name="security",
            version=VERSION,
            description="Security evaluation data with contextual security attacks"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "safety"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        base_dir = Path(__file__).parent / "data"
        
        # Determine which data directory based on config
        data_type = f"{self.config.name}_data"  # "safety_data" or "security_data"
        
        return [
            datasets.SplitGenerator(
                name="with_defense",
                gen_kwargs={
                    "manifest_file": base_dir / data_type / "with_defense" / "attack_manifest.jsonl",
                    "trace_dir": base_dir / data_type / "with_defense" / "traces",
                },
            ),
            datasets.SplitGenerator(
                name="without_defense",
                gen_kwargs={
                    "manifest_file": base_dir / data_type / "without_defense" / "attack_manifest.jsonl",
                    "trace_dir": base_dir / data_type / "without_defense" / "traces",
                },
            ),
        ]

    def _generate_examples(self, manifest_file, trace_dir):
        """Yields examples."""
        manifest_file = Path(manifest_file)
        trace_dir = Path(trace_dir)
        
        with open(manifest_file, 'r') as f:
            for idx, line in enumerate(f):
                snapshot = json.loads(line)
                trace_id = snapshot.get('result', {}).get('trace_id', '')
                
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
                        return {k: (None if (isinstance(v, dict) and not v) else v) 
                                for k, v in cleaned.items()}
                    elif isinstance(obj, list):
                        return [clean_empty_dicts(item) for item in obj]
                    return obj
                
                cleaned_snapshot = clean_empty_dicts(snapshot)
                cleaned_traces = clean_empty_dicts(traces)
                
                yield idx, {
                    "trace_id": trace_id,
                    "attack_snapshot": cleaned_snapshot,
                    "trace": cleaned_traces,
                }

