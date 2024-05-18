import os
import random
from datetime import datetime 
import pandas as pd
from pathlib import Path
import argparse
from typing import List

# Paths
PRD_ROOT    = Path('/grand/projects/SuperBERT/foster')
RAW_DB_PATH = Path('/grand/projects/SuperBERT/siebenschuh/prd_selection/')

def sample_random_zips(import_dir, n_dir:int=25, n_zip_per_dir:int=1, prd_root:Path= Path('/grand/projects/SuperBERT/foster')) -> List[Path]:
    '''Samples n_dir * n_zip_per_dir (pseudo-random) zip directories each containing ~1000k PDFs
    '''
    assert all([d in os.listdir(prd_root) for d in ['imports', 'imports2']]), "`imports1` and `imports2` must reside in `prd_root`"
    
    if import_dir is None:
        raise ValueError('`import_dir` must be integer either `1` or `2`')
    elif import_dir==1:
        p_prd = PRD_ROOT / 'imports'
    else:
        p_prd = PRD_ROOT / 'imports2'

    # check path validity
    assert p_prd.is_dir(), "Assume the path `p_prd` exists"
    
    # load
    prd_dirs = [p_prd / d for d in os.listdir(p_prd)]
    
    # pseudo-random subset of dirs
    random.seed(17)
    random.shuffle(prd_dirs)
    prd_dirs = prd_dirs[:n_dir]
    
    # sample random zips from dirs
    glob_zip_list = []
    for j,p in enumerate(prd_dirs):
        # sample zips
        zip_list = [p / d_i for d_i in os.listdir(p) if d_i.endswith('.zip')]
    
        # duplicate
        random.seed(j)
        random.shuffle(zip_list)
        
        # extract zip paths
        glob_zip_list += zip_list[:n_zip_per_dir]
    
    return glob_zip_list

def store_zip_paths(zips:list, store_df_path:Path=RAW_DB_PATH) -> None:
    '''Stores the zip paths into a pandas DataFrame
    '''
    
    assert store_df_path.is_dir(), "Path to store data `store_df_path` must be in."
    
    valid_zips = []
    for zip in zips:
        if zip.is_file():
            valid_zips.append(zip)

    # assemble DF
    df = pd.DataFrame({'path' : valid_zips}, index=None)
    
    # store 
    date_str = datetime.now().strftime('%d-%m-%Y')
    
    # file path
    df_store_path = store_df_path / ('zips_' + date_str + '.csv')
    # - no duplicate conflict
    if df_store_path.is_file():
        df_store_path = store_df_path / ('zips_' + date_str + f'_{str(random.choice(range(100))).zfill(3)}.csv')
    
    # store
    df.to_csv(df_store_path, index=None)
    
    # debug
    print(f'Stored to ... {df_store_path}')
    
    pass

def main():
    # parser
    parser = argparse.ArgumentParser(description="Assemble list of `.zip` files that are to be extracted from PRD (on eagle!)")
    
    parser.add_argument('-n', '--n_dir', type=int, default=25, help='Number of directories to crawl (each containing several zip files)')
    parser.add_argument('-z', '--n_zip_per_dir', type=int, default=2, help='Number zip files PER directory to sample')
    parser.add_argument('-i', '--prd_dir', type=Path, default=PRD_ROOT, help='Directory in which PRD`s `import` and `imports2` reside.')
    parser.add_argument('-p', '--store_df_dir', type=Path, default=RAW_DB_PATH, help='Directory to which CSV will be written')

    # parse arguments
    args = parser.parse_args()

    n_dir = args.n_dir
    n_zip_per_dir = args.n_zip_per_dir
    prd_dir = args.prd_dir
    store_df_dir = args.store_df_dir
    
    # sample zip paths 
    zips_1 = sample_random_zips(1, n_dir=n_dir, n_zip_per_dir=n_zip_per_dir, prd_root=prd_dir)
    zips_2 = sample_random_zips(2, n_dir=n_dir, n_zip_per_dir=n_zip_per_dir, prd_root=prd_dir)

    # store
    store_zip_paths(zips=zips_1+zips_2, store_df_path=store_df_dir)

    pass