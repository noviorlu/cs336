import torch
import argparse
import os
import yaml
from cs336_basics.transformer import TransformerLM, build_attention_mask
from cs336_basics.bpe_tokenizer import Tokenizer

def generate(model, input_ids, max_new_tokens, context_length, temperature=1.0, top_p=None, device='cuda', eos_id=None):
    """
    自回归文本生成 (Autoregressive Text Generation)
    """
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截断输入，确保不超过模型的最大上下文长度
            if input_ids.size(1) > context_length:
                input_ids_cropped = input_ids[:, -context_length:]
            else:
                input_ids_cropped = input_ids
                
            # Build attention mask for the cropped sequence
            seq_len = input_ids_cropped.size(1)
            mask = build_attention_mask(seq_len, device=device)
            
            # 前向传播，拿到所有的 logits
            logits = model(input_ids_cropped, mask=mask)
            
            # Step 1: 只有最后一行有用
            # 取出序列最后一个 token 对 "下一个位置" 的预测
            next_token_logits = logits[:, -1, :]
            
            # Step 2: 温度调节 (Temperature Scaling)
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            
            # Step 3: Top-p (Nucleus) Sampling 智能旋转门
            if top_p is not None and top_p < 1.0:
                # 1. 对 logits 进行降序排序
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                # 2. 算概率
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                # 3. 算累计概率 (cumsum)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 4. 剔除烂词 (处理 差一错误 Off-by-one)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 # 永远保留概率最大的第一个词！
                
                # 把排好序的剔除标记，还原回原本词表的顺序
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                
                # 斩杀烂词！把它们的 logits 设为负无穷大
                next_token_logits[indices_to_remove] = float('-inf')
                
            # 把处理好的 logits 变成概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Step 4: 抽签 (Sampling)
            if temperature == 0.0:
                # 绝对贪心，只选最大的
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # 根据概率分布掷骰子抽卡
                next_token = torch.multinomial(probs, num_samples=1)
                
            # 把抽到的新词拼接到原句子的末尾，进入下一个循环！
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 提前结束
            if eos_id is not None and next_token.item() == eos_id:
                break
                
    return input_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--config", type=str, default="tinystories_17M.yaml", help="Path to config yaml")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p cutoff (1.0 to disable)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading tokenizer...")
    vocab_path = os.path.join("data", "tinystories_vocab.json")
    merges_path = os.path.join("data", "tinystories_merges.json")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("Warning: Tokenizer files not found at data/. Make sure they exist.")
    
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    
    print(f"Loading model from {args.checkpoint}...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config.get('d_ff', config['d_model'] * 4),
        # 消融跑出来的 checkpoint 必须按它自己的架构重建。no_rope 尤其危险：
        # 去掉 RoPE 不改变任何权重的形状，state_dict 能原样装进一个带 RoPE 的
        # 模型里，不报错，只是生成一堆乱码。norm / ffn 至少还会 shape 不匹配。
        rope_theta=None if config.get('no_rope') else config.get('rope_theta', 10000.0),
        norm=config.get('norm', 'pre'),
        ffn=config.get('ffn', 'swiglu'),
    ).to(args.device)
    
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    print("Generating...\n" + "="*50)
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)

    # handout 要的是「至少 256 个 token，或者到第一个 <|endoftext|> 为止」。
    # generate() 早就支持 eos_id 提前停，之前只是没人传进去。
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    # Generate
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=config['context_length'],
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        eos_id=eos_id,
    )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    print(output_text)
    print("="*50)

if __name__ == "__main__":
    main()
