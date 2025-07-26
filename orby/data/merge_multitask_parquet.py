import datasets
from datasets import Sequence, Image


def fix_dataset_format_for_merge(dataset):
    if "images" in dataset.features:
        # Check if images are in Dataset 2 format (list of dicts with bytes/path)
        # Dataset subtask_direct_distill format: [{'bytes': binary, 'path': int32}]
        # Dataset uground,os_atlas format: Sequence(feature=Image(mode=None, decode=True))
        if type(dataset.features["images"]) == list:
            print(
                "Converting image format from [{'bytes': binary, 'path': int32}] to Sequence[Image]"
            )
            # Cast the images column to the new Sequence type
            dataset = dataset.cast_column(
                "images", Sequence(feature=Image(decode=True), length=-1)
            )

    # In all datasets, the response field is a list of dicts with role and content keys
    # The order of keys is "role" and "content" in subtask_direct_distill dataset, and "content" and "role" in uground,os_atlas dataset
    # This leads to a mismatch in the response schema between the two datasets, which results in a failure to concatenate the two datasets
    # Change response message key order for subtask_direct_distill dataset
    if (
        "data_source" in dataset.features
        and dataset["data_source"][0] == "subtask_direct_distill"
    ):
        print("Standardizing response message key order")

        # Explicitly cast the response column to ensure schema compatibility
        response_features = [
            {"content": datasets.Value("string"), "role": datasets.Value("string")}
        ]

        # Create new features with the correct order
        new_features = dataset.features.copy()
        new_features["response"] = response_features

        # Cast the dataset to the new schema
        dataset = dataset.cast(new_features)

    return dataset


parquet_files = [
    "/root/subtask/osatlas_test_part_0000.parquet",
    "/root/subtask/osatlas_test_part_0001.parquet",
    #"/root/subtask/osatlas_test_part_0006.parquet",
    #"/root/subtask/osatlas_test_part_0007.parquet",
    "/root/subtask/test_part_0000.parquet",
    "/root/subtask/test_part_0001.parquet",
    #"/root/subtask/test_part_0006.parquet",
    #"/root/subtask/test_part_0007.parquet",
    "/root/subtask/subtask_test.parquet",
]
dataframes = []
for parquet_file in parquet_files:
    dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    dataframe = fix_dataset_format_for_merge(dataframe)
    dataframes.append(dataframe)
    print(parquet_file)
    print(dataframe)

dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
print('Final', dataframe)
dataframe.to_pandas().to_parquet("/root/subtask/test.parquet", row_group_size=512)
