from transformers import AutoModelForCausalLM
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="google/gemma-2b")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    print(model.config.vocab_size)