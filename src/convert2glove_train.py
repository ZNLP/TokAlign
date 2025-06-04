import datasets
import argparse

def convert2train(
    src_path = "llama-3-tok-20GB_tk",
    tgt_path = "llama-3-tok-20GB_tk-train-GloVe",
    key = "train",
    min_line_len = 15,
    max_line = 1000000
):
    d = datasets.load_from_disk(src_path)[key]

    with open(tgt_path, "w") as f:
        for i in range(min(max_line, len(d))):
            data = d[i]["input_ids"]
            if len(data) < min_line_len:
                continue
            line = " ".join([f"{cd}" for cd in data])
            f.write(line+"\n")

def convert2eval(
    src_tok_path = "llama-3-tok-20GB_tk",
    tgt_tok_path = "gemma-tok-20GB_tk",
    file_path = "llama3-2-gemma-MX1k-eval",
    key = "validation",
    min_line_len = 15,
    max_line = 1000,
):
    d1 = datasets.load_from_disk(src_tok_path)
    d2 = datasets.load_from_disk(tgt_tok_path)
    assert(len(d1[key]) == len(d2[key]))

    with open(file_path, "w") as f:
        for i in range(max_line):
            data1 = d1[key][i]["input_ids"]
            data2 = d2[key][i]["input_ids"]
            if len(data1) < min_line_len or len(data2) < min_line_len:
                continue

            line1 = " ".join([f"{cd}" for cd in data1])
            line2 = " ".join([f"{cd}" for cd in data2])

            f.write(f"{line1}\t{line2}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", type=str, default="./data/pretrain-dataset/multilingual-pythia-tok_tk")
    parser.add_argument("-t", "--target-path", type=str, default="./data/pretrain-dataset/multilingual-llama-tok_tk")
    parser.add_argument("-k", "--key", type=str, default="train")
    parser.add_argument("-m", "--min-line-length", type=int, default=15)
    parser.add_argument("-l", "--max-line", type=int, default=10000000)
    parser.add_argument("-o", "--output-path", type=str, default="./data/pretrain-dataset/pythia-tok_train-GloVe")

    args = parser.parse_args()

    if args.key == "train":
        convert2train(
            src_path = args.source_path,
            tgt_path = args.output_path,
            key = "train",
            min_line_len = args.min_line_length,
            max_line = args.max_line
        )
    elif args.key == "valid" or args.key == "validation":
        convert2eval(
            src_tok_path = args.source_path,
            tgt_tok_path = args.target_path,
            file_path = args.output_path,
            key = args.key,
            min_line_len = args.min_line_length,
            max_line = args.max_line,
        )
    else:
        raise Exception(f"Method of {args.key} is not implemented.")

    # convert2train(
    #     src_path = "./data/pretrain-dataset/multilingual-gemma-tok_tk",
    #     tgt_path = "./data/pretrain-dataset/gemma-tok_train-GloVe",
    #     key = "train",
    #     min_line_len = 15,
    #     max_line = 10000000
    # )

    # convert2train(
    #     src_path = "./data/pretrain-dataset/multilingual-pythia-tok_tk",
    #     tgt_path = "./data/pretrain-dataset/pythia-tok_train-GloVe",
    #     key = "train",
    #     min_line_len = 15,
    #     max_line = 10000000
    # )
