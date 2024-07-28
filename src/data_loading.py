import fiftyone as fo

def load_datasets():
    real_dataset_1 = fo.load_dataset('DIMS_REAL_ICMU_BODYKEYPOINT2D_002')
    real_dataset_2 = fo.load_dataset('DIMS_REAL_ICMU_SEATBELT_001')
    syn_dataset = fo.load_dataset('DIMS_SYN_ICMU_OBDBKPAGEOCL_000')
    return real_dataset_1, real_dataset_2, syn_dataset

def merge_datasets(real_dataset_1, real_dataset_2):
    merged_real_dataset = fo.Dataset()
    merged_real_dataset.merge_samples(real_dataset_1)
    merged_real_dataset.merge_samples(real_dataset_2)
    return merged_real_dataset
