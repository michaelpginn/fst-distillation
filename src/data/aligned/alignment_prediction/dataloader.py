from torch.utils.data import DataLoader

from src.modeling.input_processing import create_pad_mask, pad_batch_collate_fn

from .dataset import AlignmentPredictionDataset


def create_dataloader(
    dataset: AlignmentPredictionDataset,
    batch_size: int,
):
    """Dataloader for alignment prediction task

    Arguments:
        data_path: str, path to an aligned inflection file
        batch_size: int
    """
    tokenizer = dataset.tokenizer

    def collate_fn(batch: list[dict[str, int | list[int]]]):
        collated_batch = pad_batch_collate_fn(batch, tokenizer.pad_token_id)
        collated_batch["source_pad_mask"] = create_pad_mask(
            collated_batch["source_input_ids"], tokenizer.pad_token_id
        )
        if "target_input_ids" in collated_batch:
            collated_batch["target_pad_mask"] = create_pad_mask(
                collated_batch["target_input_ids"], tokenizer.pad_token_id
            )
        return collated_batch

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    return dataloader
