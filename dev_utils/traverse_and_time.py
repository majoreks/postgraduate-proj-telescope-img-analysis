import time
from torch.utils.data import DataLoader
from tqdm import tqdm

def traverse_and_time(loader: DataLoader, times: int = 5) -> float:
    start_time = time.time()
    
    for _ in tqdm(range(times), total=times, desc="dataloader loop"):
        for _, (_, _) in tqdm(enumerate(loader), total=len(loader), desc="single dataloader"):
            pass
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time to traverse dataset {times} times: {elapsed:.3f} seconds")

    return elapsed