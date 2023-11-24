import os
import torch
import re

from typing import List,Dict,Literal
from pathlib import Path
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm


class ConfigDataSet(BaseModel):
        split: Literal['train','dev','test']
        model_name: Literal['gpt2-large','gpt2-medium','gpt2','gpt2-xl'] = 'gpt2'
        trun_limit: int = 500    
        BASEPATH : Path = Path("../data/")


class EnronEmailDataset(Dataset):
    # Read About MRO(Method Resolution Order)    
    
    def __init__(
        self,
        config: ConfigDataSet
    ):
        # As Config is at as just data we can us it with pydatic
        self.config = config
        self.tokenizer  = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.file_paths: List[str] = [ self.config.BASEPATH/self.config.split/name 
                                      for name in 
                                      os.listdir(self.config.BASEPATH/self.config.split)]
        self.emails: List[str] = [ open(self.file_paths[idx],'r').read().strip()\
                                         for idx in tqdm(range(len(self.file_paths)),desc="Loading Email") ]
        
    def clean_text(
        self,
        text:str
        ):
        # Updated so that it comes from config
        # ipdb.set_trace()
        text = re.sub(' +',' ',text)
        text = re.sub('\n+','\n',text)
        text = re.sub('[^A-Za-z0-9\n\s\\/.-]+','',text)
        return text
        
        
    def __getitem__(
        self,
        idx:int
    ):
        
        """ 
        returns the input_ids and attention_maks also tuncates if
        the email is longer that what is specified in config
        """
        
        # with open(self.file_paths[idx],'r') as f:
        #     email_with_subject = f.read().strip()
        
        email_with_subject = self.emails[idx]

        email,subject = email_with_subject.split("@subject\n")
        
        email = self.clean_text(email)

        email = ''.join(email.splti()[:self.config.trun_limit])

        return self.tokenizer(email +"@subject\n"+ subject)


        # tok_email = self.tokenizer(email,truncation=True,max_length=self.config.trun_limit)
        # tok_subject = self.tokenizer( "@subject\n"+ subject + " <|endoftext|>",
        #                              truncation=True,max_length=self.config.trun_limit)
        
        # Token from which CLM will start Finetuning
        # st_gen_token = len(tok_email['input_ids'])
        
        # tok_email['input_ids'].extend(tok_subject['input_ids'])
        # tok_email['attention_mask'].extend([0]*len(tok_subject['attention_mask']))
        
        
         # return ({'input_ids':torch.tensor(tok_email['input_ids']),
         #         "attention_mask":torch.tensor(tok_email['attention_mask'])},st_gen_token)

        
    def __len__(
        self
    ):
        return len(self.file_paths)


if __name__=="__main__":
        dataconfig = ConfigDataSet(split='train',
                                   trun_limit=500)
        
        dataset = EnronEmailDataset(dataconfig)
