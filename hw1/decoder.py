import torch
import argparse
import os
import yaml
from cs336_basics.model import TransformerLM
from cs336_basics.bpe_tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--config", type=str, default="tinystories_17M.yaml", help="Path to config yaml")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p cutoff (1.0 to disable)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="词表前缀（data/<前缀>_vocab.json）。不给就按 vocab_size 自动选："
                             "10000→tinystories，32000→owt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading tokenizer...")
    # 词表默认按 vocab_size 认：10000 是 TinyStories，32000 是 OpenWebText。
    # 写死 tinystories 的话，拿 OWT 的 checkpoint 生成会用错词表——id 对不上，
    # 出来是乱码，而且不报错。
    stem = args.tokenizer or ("owt" if config['vocab_size'] > 20000 else "tinystories")
    vocab_path = os.path.join("data", f"{stem}_vocab.json")
    merges_path = os.path.join("data", f"{stem}_merges.json")
    print(f"tokenizer: {stem}（vocab_size={config['vocab_size']}）")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("Warning: Tokenizer files not found at data/. Make sure they exist.")
    
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    
    print(f"Loading model from {args.checkpoint}...")
    model = TransformerLM.from_pretrained(args.checkpoint, device=args.device)
    
    print("Generating...\n" + "="*50)
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)

    # handout 要的是「至少 256 个 token，或者到第一个 <|endoftext|> 为止」。
    # generate() 早就支持 eos_id 提前停，之前只是没人传进去。
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    # Generate
    output_ids = model.generate(
        x=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    print(output_text)
    print("="*50)

if __name__ == "__main__":
    main()
