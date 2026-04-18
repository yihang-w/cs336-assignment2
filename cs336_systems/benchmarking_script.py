import torch
import timeit
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy



def benchmarking_script(
                        batch_size:int,
                        dataset_path:str | Path,
                        vocab_size: int,
                        context_length: int,
                        d_model: int,
                        num_layers: int,
                        num_heads: int,
                        d_ff: int,
                        rope_theta: float,
                        warm_steps:int,
                        benchmarking_steps:int,
                        device:str,
                        backward:bool = False,
                        ) -> None:
    #准备数据
    dataset = np.load(dataset_path,mmap_mode='r')
    data,label = get_batch(dataset,batch_size,context_length,device)
    
    #准备模型
    model = BasicsTransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta).to(device)
    
    #预热
    for i in range(warm_steps):
        y = model(data)
        if backward:
            loss = cross_entropy(y,label)
            loss.backward()
            
    time_data = pd.DataFrame({"forward":[0.0] *benchmarking_steps,"backward":[0.0] *benchmarking_steps}) if backward else pd.DataFrame({"forward":[0.0] *benchmarking_steps})
    for i in range(benchmarking_steps):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        y = model(data)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        time_data.loc[i,'forward'] = end - start
        if backward:
            loss = cross_entropy(y,label)
            torch.cuda.synchronize()
            start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            time_data.loc[i,"backward"] = end - start
    results = time_data.agg(('mean',"std"))
    print(time_data.to_markdown())
    print(results.to_markdown())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("batch_size",type=int)
    parser.add_argument('dataset_path',type=str)
    parser.add_argument("vocab_size",type=int)
    parser.add_argument("context_length",type=int)
    parser.add_argument("d_model",type=int)
    parser.add_argument("num_layers",type=int)
    parser.add_argument("num_heads",type=int)
    parser.add_argument("d_ff",type=int)
    parser.add_argument("rope_theta",type=float)
    parser.add_argument("warm_steps",type=int)
    parser.add_argument("benchmarking_steps",type=int)
    parser.add_argument("--backward",type=bool,default=False)
    parser.add_argument("device",type=str)
    args = parser.parse_args()           
    benchmarking_script(**vars(args))
    