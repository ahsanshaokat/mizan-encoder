import json
import random

def load_sts_tsv(path, sample_size=None):
    """Load STS dataset from TSV file"""
    pairs = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                score = float(parts[4]) / 5.0  # Normalize to [0, 1]
                pairs.append((parts[5], parts[6], score))
        
        if sample_size and len(pairs) > sample_size:
            pairs = random.sample(pairs, sample_size)
            
    except Exception as e:
        print(f"Error loading STS data: {e}")
        return []
    
    return pairs


def load_snli_jsonl(path, sample_size=None):
    """Load SNLI dataset from JSONL file"""
    pairs = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['gold_label'] == 'entailment':
                    # Treat entailment as positive pairs
                    pairs.append((data['sentence1'], data['sentence2'], 1.0))
                elif data['gold_label'] == 'contradiction':
                    # Treat contradiction as negative pairs
                    pairs.append((data['sentence1'], data['sentence2'], 0.0))
        
        if sample_size and len(pairs) > sample_size:
            pairs = random.sample(pairs, sample_size)
            
    except Exception as e:
        print(f"Error loading SNLI data: {e}")
        return []
    
    return pairs


class PairDataset(torch.utils.data.Dataset):
    """Dataset for sentence pairs"""
    
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text1, text2, score = self.pairs[idx]
        return text1, text2, score