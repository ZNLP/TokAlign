import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import itertools

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=4096,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
        start_idx=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        if start_idx != 0:
            print(f"Reset the start index of dataset to {start_idx}.")
            self.dataset = concatenate_datasets([dataset.select(range(start_idx, len(dataset))), dataset.select(range(0, start_idx))])
        self.need_tokenize = False if 'input_ids' in dataset.features else True
        self.content_field = 'input_ids' if 'input_ids' in dataset.features else content_field
        self.seq_length = seq_length
        # print(f"Current sequence length is: {seq_length}")
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token
    
    def direct_iter(self):
        iterator = iter(self.dataset)
        while True:
            buffer, buffer_len = [], 0

            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    sample = next(iterator)[self.content_field]
                    assert len(sample) == self.seq_length
                    buffer.append(sample)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            examples = buffer
            
            # if self.shuffle:
            #     random.shuffle(examples)

            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

    def token_iter(self):
        iterator = iter(self.dataset)
        while True:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

    def sample_mapper(self, sample):
        return {
            "input_ids": torch.LongTensor(sample["input_ids"]),
            "labels": torch.LongTensor(sample["input_ids"]),
        }
    
    def multiprocessing_iter(self):
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        iterator = iter(self.dataset)

        return itertools.islice(map(self.sample_mapper, iterator), worker_id, None, worker_total_num)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is None:
            return self.token_iter() if self.need_tokenize else self.direct_iter()
        else:
            return self.multiprocessing_iter()
def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column])['input_ids'])
        # total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def create_datasets(tokenizer, args):
    # dataset = load_dataset(args.dataset_name, use_auth_token=True, num_proc=args.num_workers)
    dataset = load_from_disk(args.dataset_name)
    train_data = dataset["train"]
    valid_data = dataset["test"] if "test" in dataset else dataset["validation"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field) if 'text' in train_data.features else 1
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=True,
        add_eos_token=False,
        start_idx=args.train_start_idx,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=False,
        add_eos_token=False,
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    if args.use_flash_attn:
        warnings.warn(
            "Flash V2 support implemented here ignores padding/attention_mask/custom_mask. \n"
            + "It is meant for continued pre-training with packing inputs to consume the entire sequence lengths."
        )
        # from starcoder_flash_attn_monkey_patch import replace_starcoder_attn_with_flash_attn
        from llama_flash_attn_patch import replace_llama_attn_with_flash_attn
        # from falcon_flash_attn_monkey_patch import replace_falcon_attn_with_flash_attn

        # replace_starcoder_attn_with_flash_attn()
        replace_llama_attn_with_flash_attn()
        # replace_falcon_attn_with_flash_attn()
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path is not None else args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model.resize_token_embeddings(len(tokenizer))


    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
