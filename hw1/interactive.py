import torch
import argparse
import os
import yaml
from cs336_basics.transformer import TransformerLM, build_attention_mask
from cs336_basics.bpe_tokenizer import Tokenizer
from decoder import generate

def main():
    parser = argparse.ArgumentParser()
    # 核心配置文件
    parser.add_argument("--gen_config", type=str, default="generate_config.yaml", help="Path to generation config yaml")
    
    # 允许在命令行覆盖 yaml 里的配置（可选）
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Override model config path")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Override top-p")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    # 1. Load Generation Config (读取生成专用的 YAML)
    print(f"Loading generation config from {args.gen_config}...")
    with open(args.gen_config, "r") as f:
        gen_config = yaml.safe_load(f)
        
    # 合并命令行参数（如果命令行指定了，就覆盖 yaml 里的）
    checkpoint_path = args.checkpoint if args.checkpoint is not None else gen_config['checkpoint']
    model_config_path = args.config if args.config is not None else gen_config['model_config']
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else gen_config['max_new_tokens']
    temperature = args.temperature if args.temperature is not None else gen_config['temperature']
    top_p = args.top_p if args.top_p is not None else gen_config['top_p']
    device = args.device if args.device is not None else gen_config.get('device', 'cuda')

    print("\n" + "="*50)
    print("🤖 正在启动 TinyStories 交互终端...")
    print("="*50)

    # 2. Load Model Config (读取模型结构的 YAML)
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Load Tokenizer
    vocab_path = os.path.join("data", "tinystories_vocab.json")
    merges_path = os.path.join("data", "tinystories_merges.json")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("Warning: Tokenizer files not found at data/. Make sure they exist.")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    
    # 3. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config.get('d_ff', config['d_model'] * 4),
        rope_theta=config.get('rope_theta', 10000.0)
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    print("\n✅ 模型加载完毕！")
    print("输入 'quit' 或 'exit' 退出程序，输入 'clear' 清空控制台。")
    print("注意：对于 TinyStories，最好输入诸如 'Once upon a time, ' 这样的童话开头。")
    print("="*50 + "\n")
    
    # 4. Interactive Loop (支持上下文接龙)
    history_ids = None
    
    while True:
        try:
            prompt = input("\033[92m[Prompt]: \033[0m")
            if prompt.strip() == "":
                continue
            if prompt.strip().lower() in ['quit', 'exit']:
                print("Bye! 👋")
                break
            if prompt.strip().lower() == 'clear':
                os.system('clear')
                history_ids = None
                print("🧹 上下文历史已清空！")
                continue
                
            # 将用户的输入编码
            prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            
            # 拼接到历史记录上
            if history_ids is None:
                input_ids = prompt_ids
            else:
                input_ids = torch.cat([history_ids, prompt_ids], dim=1)
            
            # 如果加上新的 prompt 已经超过了模型的最大上下文长度，强制截断
            ctx_len = config['context_length']
            if input_ids.size(1) > ctx_len:
                input_ids = input_ids[:, -ctx_len:]
                print(f"\033[93m[警告]: 对话过长，已自动截断至最近的 {ctx_len} 个 Token。\033[0m")
            
            # 生成后续文本
            output_ids = generate(
                model, 
                input_ids, 
                max_new_tokens=max_new_tokens, 
                context_length=config['context_length'],
                temperature=temperature, 
                top_p=top_p, 
                device=device,
                eos_id=tokenizer.encode("<|endoftext|>")[0]
            )
            
            # 把历史记录更新为完整的序列
            history_ids = output_ids
            
            # 我们只把“刚刚新生成”的那部分 Token 拿出来解码并高亮显示
            new_generated_ids = output_ids[0][input_ids.size(1):].tolist()
            new_text = tokenizer.decode(new_generated_ids)
            
            # 打印模型新生成的续写内容
            print("\033[96m[Model]: \033[0m", end="")
            print(new_text)
            print(f"\n\033[90m(本次生成了 {len(new_generated_ids)} 个 Token)\033[0m")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nBye! 👋")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()
