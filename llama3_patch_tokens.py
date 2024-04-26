from transformers import AutoTokenizer

import os
import re

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
from tokenizers import AddedToken, pre_tokenizers
from transformers import AutoTokenizer
#pre_tokenizers.ByteLevel(False,False).pre_tokenize_str("Bác")

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#tokenizer.add_tokens(AddedToken("BÃ¡c", normalized=False,special=False))
#>>> tokenizer.decode(tokenizer.encode("Bác"))

tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
super_sp_model = spm.SentencePieceProcessor()
super_sp_model.Load('2228_tokens_100m.model')


super_spm = sp_pb2_model.ModelProto()
super_spm.ParseFromString(super_sp_model.serialized_model_proto())

  
def is_russian(s: str) -> bool:
    # Данное регулярное выражение соответствует строке, которая содержит только кириллические символы,
    # пробелы, тире и знаки препинания. Всё остальное будет считаться недопустимым.
    return bool(re.match("^[а-яА-ЯёЁ\s.,!?:;-_]*$", s))
z = []
for p in super_spm.pieces:
    if is_russian(p.piece):
        z+=[p.piece]
from tqdm import tqdm
for zs in tqdm(z):
    
    tokenizer.add_tokens(AddedToken(zs, normalized=False,special=False))
tokenizer.save_pretrained('llama_patched')
#llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
#print(len(llama_spm_tokens_set))
#print(f"Before: {len(llama_spm_tokens_set)}")
