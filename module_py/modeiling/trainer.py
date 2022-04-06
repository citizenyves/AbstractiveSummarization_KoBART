import os, sys
import datetime
import transformers
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed
from data.dataset import Collator
from data.make_input import train_dataset, valid_dataset

# set seed
set_seed(config)

# model & tokenizer
model = transformers.BartForConditionalGeneration.from_pretrained(config.pretrained_model_name)
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

def Trainer(model, tokenizer):
    # training arguments
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch", 
        per_device_train_batch_size=config.batch_size, 
        per_device_eval_batch_size=config.batch_size,  
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False, #gpu 사용시 True
        dataloader_num_workers=4,
        disable_tqdm=False,
        load_best_model_at_end=True,                 
        
        ## As below, only Seq2SeqTrainingArguments' arguments.
        # sortish_sampler=True,                     # Whether to use a `sortish sampler` or not.
        # predict_with_generate=True,               # Whether to use generate to calculate generative metrics (ROUGE, BLEU)
        generation_max_length=config.tar_max_len,   # 150
        generation_num_beams=config.beam_size,      # 1
    )

    # Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=Collator(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            inp_max_len=config.inp_max_len,
            tar_max_len=config.tar_max_len,
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )    
    
    return trainer