import fiftyone as fo

def load_datasets():
    real_dataset_1 = fo.load_dataset('REAL_DATASET_001')
    real_dataset_2 = fo.load_dataset('REAL_DATASET_002')
    syn_dataset = fo.load_dataset('SYN_DATASET_001')
    return real_dataset_1, real_dataset_2, syn_dataset

def merge_datasets(real_dataset_1, real_dataset_2):
    merged_real_dataset = fo.Dataset()
    merged_real_dataset.merge_samples(real_dataset_1)
    merged_real_dataset.merge_samples(real_dataset_2)
    return merged_real_dataset
