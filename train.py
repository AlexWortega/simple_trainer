#from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli
import math
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def freeze(
    model,
    freeze_emb=False,
    freeze_ln=True,
    freeze_attn=True,
    freeze_ff=True,
    freeze_ff_layers=None,  # None means all or no layers, depending on freeze_ff
    freeze_other=False,
):
    if freeze_ff_layers is not None and not isinstance(freeze_ff_layers, (list, set)):
        raise ValueError("freeze_ff_layers must be a list or set of layer indices")

    for name, p in model.named_parameters():
        name = name.lower()
        layer_index = None
        if 'mlp' in name:
            # Parse the layer index from the parameter name if possible
            tokens = name.split('.')
            for token in tokens:
                if token.isdigit():
                    layer_index = int(token)
                    break
        
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            if freeze_ff_layers is None:
                # Apply general freeze_ff setting
                p.requires_grad = not freeze_ff
            else:
                # Apply specific layer freeze setting
                p.requires_grad = not (freeze_ff and layer_index in freeze_ff_layers)
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            print(p.name)
            p.requires_grad = not freeze_other
    return model



from transformers import AutoTokenizer, AutoModelForCausalLM
def main():
    import torch
    from datasets import Dataset
    #import tensor_parallel as tp
    tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-7b-0.1")
    
    model = AutoModelForCausalLM.from_pretrained("Vikhrmodels/Vikhr-7b-0.1")

    #model.resize_token_embeddings(len(tokenizer))
    # device_map="auto")
    #model = tp.tensor_parallel(model, ["cuda:0", "cuda:1","cuda:2"])
    
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    from datasets import load_dataset
    tokenizer.pad_token = tokenizer.eos_token
    
    from datasets import load_dataset
    #datasets.load_dataset('dichspace/darulm', domains=["habr","textbook"], split="train", streaming=True):
        _URLS = [
        "libru_accounting_0.jsonl.zst",
        "libru_antique_0.jsonl.zst",
        "libru_antique_1.jsonl.zst",
        "libru_aphorisms_0.jsonl.zst",
        "libru_art_0.jsonl.zst",
        "libru_biography_0.jsonl.zst",
        "libru_biography_1.jsonl.zst",
        "libru_biography_2.jsonl.zst",
        "libru_biography_3.jsonl.zst",
        "libru_biography_4.jsonl.zst",
        "libru_biology_0.jsonl.zst",
        "libru_business_0.jsonl.zst",
        "libru_cinema_0.jsonl.zst",
        "libru_computers_0.jsonl.zst",
        "libru_design_0.jsonl.zst",
        "libru_dramaturgy_0.jsonl.zst",
        "libru_economics_0.jsonl.zst",
        "libru_essay_0.jsonl.zst",
        "libru_essay_1.jsonl.zst",
        "libru_essay_2.jsonl.zst",
        "libru_fantasy_0.jsonl.zst",
        "libru_geography_0.jsonl.zst",
        "libru_guidebooks_0.jsonl.zst",
        "libru_guidebooks_1.jsonl.zst",
        "libru_history_0.jsonl.zst",
        "libru_history_1.jsonl.zst",
        "libru_history_2.jsonl.zst",
        "libru_history_3.jsonl.zst",
        "libru_history_4.jsonl.zst",
        "libru_history_5.jsonl.zst",
        "libru_humor_0.jsonl.zst",
        "libru_language_0.jsonl.zst",
        "libru_law_0.jsonl.zst",
        "libru_literature_0.jsonl.zst",
        "libru_medicine_0.jsonl.zst",
        "libru_military_0.jsonl.zst",
        "libru_music_0.jsonl.zst",
        "libru_philosophy_0.jsonl.zst",
        "libru_politic_0.jsonl.zst",
        "libru_prose_0.jsonl.zst",
        "libru_prose_1.jsonl.zst",
        "libru_prose_2.jsonl.zst",
        "libru_psychology_0.jsonl.zst",
        "libru_psychology_1.jsonl.zst",
        "libru_reference_0.jsonl.zst",
        "libru_religion_0.jsonl.zst",
        "libru_religion_1.jsonl.zst",
        "libru_religion_2.jsonl.zst",
        "libru_religion_3.jsonl.zst",
        "libru_science_0.jsonl.zst",
        "libru_science_1.jsonl.zst",
        "libru_science_2.jsonl.zst",
        "libru_sociology_0.jsonl.zst",
        "libru_textbook_0.jsonl.zst",
        "libru_UNDEFINED_0.jsonl.zst",
        "rulm_buriy_0.jsonl.zst",
        "rulm_buriy_1.jsonl.zst",
        "rulm_buriy_2.jsonl.zst",
        "rulm_gazeta_0.jsonl.zst",
        "rulm_habr_0.jsonl.zst",
        "rulm_habr_1.jsonl.zst",
        "rulm_lenta_0.jsonl.zst",
        "rulm_ods-tass_0.jsonl.zst",
        "rulm_ods-tass_1.jsonl.zst",
        "rulm_pikabu_0.jsonl.zst",
        "rulm_pikabu_1.jsonl.zst",
        "rulm_pikabu_2.jsonl.zst",
        "rulm_taiga-fontanka_0.jsonl.zst",
        "rulm_wiki_0.jsonl.zst",
        "rulm_wiki_1.jsonl.zst",
        "rulm_wiki_2.jsonl.zst",
        "wiki40_enwiki_0.jsonl.zst",
    ]
    DOMAINS = sorted(set(url.split('_')[1] for url in _URLS))
    dataset = load_dataset("dichspace/darulm",domains=DOMAINS, split="train",  cache_dir='.', streaming=True)

    
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizer
    import torch
    
    
    
    
    # Define the features of the dataset based on the iterable dataset's features
    
    
    
    # Check the dataset
    print(dataset)
    #dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, iterable_ds), features=iterable_ds.features)
    #print(dataset)
    
    import torch
    from multiprocessing import Process, Queue, Manager
    
    
    import torch
    from torch.utils.data import Dataset,DataLoader
    
    from concurrent.futures import ProcessPoolExecutor
    
    
    
    
    from torch.utils.data import Dataset
    
 
    
   
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import PreTrainedTokenizer
    
    def custom_collate_fn(batch, tokenizer: PreTrainedTokenizer, max_length=1024):
        # Extract texts from the dictionaries in the batch
        texts = [example['text'] for example in batch]
        # Tokenize all texts
        tokenized_batch = [tokenizer.encode(text, padding='max_length', max_length=max_length, truncation=True, add_special_tokens=True) for text in texts]
        tokenized_batch = torch.tensor(tokenized_batch)
        return tokenized_batch
    
    # Assuming 'dataset' is a list of dictionaries like the one you provided,
    # and 'tokenizer' is an instance of a PreTrainedTokenizer from the transformers library
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=32,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer=tokenizer),
    )


    
    
    #concatenated_dataset = rulm_Dataset(dataset, tokenizer)
    
    
    import torch
    
    
   
    
    #from torch.utils.data import DataLoader
    #loader = DataLoader(concatenated_dataset, batch_size=1, shuffle=True)
    
   
    #LL_adamw
    
    
    from transformers import  AdamW, get_linear_schedule_with_warmup
    from transformers import get_cosine_schedule_with_warmup
    
    import torch.optim as optim
    #import torch_optimizer as optim
    #optimizer = optim.NovoGrad(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_averaging=False, amsgrad=False,)
    #optimizer = ChildTuningAdamW(model.parameters(), lr = 5e-6)
    #optimizer = optim.Lamb(model.parameters(),lr= 1e-4) #betas=(0.9, 0.999),eps=1e-8,weight_decay=0,)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,total_steps=total_steps,div_factor=25, pct_start=0.2)
    lr = 3e-4 #0.0003
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-5, betas=(0.9, 0.95))
    total_steps = len(loader)

    import wandb
     #wandb.login(key = 'e461a6a3bca9f7cec3390a40dc10cdf576ce3252')
    wandb.init(name=f'tiny_llama_lr_{lr}_cl_freeze_init_128_batch', project='tiny_llama')
    
    
    
    num_warmup_steps = int(0.05 * len(loader))  # 5% of total steps
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(loader))
    
   
    
    accelerator = Accelerator(mixed_precision='fp16',gradient_accumulation_steps=128)
    device = accelerator.device
    #model = freeze(model)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    #model.to(torch.float16)
    
    model = model.to(device)
    
    model.checkpointing=True
    
    
    i = 0
    num_tokens = 0
    gradient_accumulation_steps = 128
    import time
    import gc
    # model.train()
    max_grad_norm = 1.0
    for k in range(3):
        for input_ids in tqdm(training_dataloader):
             with accelerator.accumulate(model):
                optimizer.zero_grad()
                i += 1
                start_time = time.time()
        
                input_ids = torch.cat(input_ids, 0)
                outputs = model(input_ids=input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                
                end_time = time.time()
                fw_time = end_time - start_time
        
                start_time = time.time()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
    
                end_time = time.time()
                backward_time = end_time - start_time
                optimizer.step()
                scheduler.step()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    num_tokens += 1024 * gradient_accumulation_steps

                   
                    wandb.log({
                            'loss': loss.item() * gradient_accumulation_steps,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'num_tokens': num_tokens,
                            #"eval loss":eval_loss.item()
                            #'eval loss':eval_loss.item()
                        })
    
                   
                    
                if i % 10000 == 0:
                    try:
                        state_dict = model.state_dict()
        
                        accelerator.save(state_dict, f"LL_adamw_1_tini_ru_freeze/sft_{k}_{i}")
        
                        
                    except:
                         pass
                    try:
                        accelerator.save_state(f"LL_adamw_1_tini_ru_freeze/")
                    except:
                        pass
                        
if __name__=='__main__':
    main()

