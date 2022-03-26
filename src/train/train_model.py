from src.utils.split import Split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import webdataset as wds
from src import constants
from src.model.config import Config, Input, ALL_CONFIGS
from src.model.model import Model
from src.utils.run_once import run_once
import torchvision.transforms as transforms


def identity(x):
    return x


class MyCrop(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.input == Input.AUDIO:
            random_crop = transforms.RandomCrop(
                (config.feat_seq_len, 1,), pad_if_needed=True)
        elif config.input == Input.MFCC:
            random_crop = transforms.RandomCrop(
                (config.feat_seq_len, 40,), pad_if_needed=True)
        elif config.input == Input.MFCC_EXT:
            random_crop = transforms.RandomCrop(
                (config.feat_seq_len, 120,), pad_if_needed=True)
        elif config.input == Input.XLSR:
            random_crop = transforms.RandomCrop(
                (config.feat_seq_len, 1024,), pad_if_needed=True)
        self.random_crop = random_crop

    def forward(self, x):
        out = self.random_crop(x)
        return out


class MyNormalize(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = var.sqrt()

    def forward(self, x: torch.Tensor):
        out = (x - self.mean) / self.std
        return out


def make_transform(
    config: Config,
    mean: torch.Tensor,
    var: torch.Tensor,
):
    return transforms.Compose([
        MyCrop(config),
        MyNormalize(mean, var),
    ])


def make_dataloader(config: Config, split: Split, norm_split: Split, example: bool, cpus: int):

    # For printing...
    split_name = str(split).lower().split(".")[1]
    norm_split_name = str(norm_split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Creating dataloader for {split_name} set using norm split {norm_split_name}.")

    # Select train, val, trainval or test dataset.
    dataset = constants.get_dataset(split, example)
    norm_dataset = constants.get_dataset(norm_split, example)
    file_name = str(config.input).lower().split(".")[1]  # audio, mfcc, ...

    # Load mean/var.
    mu_path = norm_dataset.norm_dir.joinpath(f"{file_name}.mu.pt")
    var_path = norm_dataset.norm_dir.joinpath(f"{file_name}.var.pt")
    if not mu_path.exists() or not var_path.exists():
        msg = f"Cannot find {file_name}.mu.pt and {file_name}.var.pt in {dataset.norm_dir}."
        raise Exception(msg)
    mean = torch.load(mu_path)
    var = torch.load(var_path)

    # Find shards.
    if split == Split.TRAINVAL:
        # Simply combine train+val shards.
        ds_train = constants.get_dataset(Split.TRAIN, example)
        ds_val = constants.get_dataset(Split.TRAIN, example)
        shards_train = ds_train.shards_dir.glob(f"{file_name}.*.tar")
        shards_train = list(map(str, shards_train))
        shards_val = ds_val.shards_dir.glob(f"{file_name}.*.tar")
        shards_val = list(map(str, shards_val))
        shard_paths = shards_train + shards_val
    else:
        shard_paths = dataset.shards_dir.glob(f"{file_name}.*.tar")
        shard_paths = list(map(str, shard_paths))
    if len(shard_paths) == 0:
        msg = f"{example_str}No shards found for {file_name}."
        raise Exception(msg)

    # Create transform.
    transform = make_transform(config, mean, var)

    # Create WebDataset (i.e. torch Dataset).
    wds_dataset = (
        wds.WebDataset(shard_paths)
        .shuffle(1000)
        .decode()
        .to_tuple("features.pth norm_mos.pth")
        .map_tuple(transform, identity)
    )
    wds_dataset = wds_dataset.batched(config.train_config.batch_size)

    # Create WebLoadert (i.e. torch DataLoader).
    wds_loader = wds.WebLoader(
        wds_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )
    # wds_loader = wds_loader.unbatched().shuffle1000).batched(config.train_config.batch_size)
    return wds_loader


def _train_model(config: Config, example: bool, bas: bool, use_subset: bool, use_trainval: bool, cpus: int, gpus: int):

    # Create model.
    model = Model(config)

    # Create dataloader(s).
    if use_subset:
        train_split = Split.TRAIN_SUBSET
        val_split = Split.VAL_SUBSET
    elif use_trainval:
        train_split = Split.TRAINVAL
        val_split = None
    else:
        train_split = Split.TRAIN
        val_split = Split.VAL
    train_dl = make_dataloader(config, train_split, train_split, example, cpus)
    if val_split is None:
        val_dl = None
    else:
        val_dl = make_dataloader(config, val_split, train_split, example, cpus)

    # Trainer parameters.
    example_name = "_example" if example else ""
    trainval_str = "_trainval" if use_trainval else "_train"
    subset_str = "_subset" if use_subset else ""
    bas_str = "_bas" if bas else ""
    out_name = f"trained_model_{config.name}{example_name}{trainval_str}{subset_str}{bas_str}"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    if val_split is None:
        raise Exception("this training type is not supported")
        best_ckpt_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath=str(model_dir),
            filename="best-{epoch:03d}-{train_loss:.6f}",
            save_top_k=3,
            mode="min",
        )
    else:
        best_ckpt_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(model_dir),
            filename="best-{epoch:03d}-{val_loss:.6f}",
            save_top_k=3,
            mode="min",
        )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="last",
    )
    trainer_params = {
        "gpus": gpus,
        "max_epochs": config.train_config.max_epochs,
        "weights_save_path": str(),
        # "strategy": "ddp",  # distributed computing
        "callbacks": [best_ckpt_callback, last_ckpt_callback],
        "progress_bar_refresh_rate": 50,
        "weights_summary": "full",
    }
    trainer = pl.Trainer(**trainer_params)

    last_ckpt_path = model_dir.joinpath("last.ckpt")
    if last_ckpt_path.exists():
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(last_ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)


def train_model(config: Config, example: bool, bas: bool, use_subset: bool, use_trainval: bool, cpus: int, gpus: int):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    trainval_str = "_trainval" if use_trainval else "_train"
    subset_str = "_subset" if use_subset else ""
    flag_name = f"trained_model_{config.name}{example_name}{trainval_str}{subset_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(config, example, bas, use_subset, use_trainval, cpus, gpus)
        else:
            print(f"{example_str}Model already trained for {config.name}.")


if __name__ == "__main__":
    example: bool = True
    bas: bool = False
    use_trainval: bool = False
    cpus: int = 1
    gpus: int = 1
    for use_subset in [True, False]:
        for config in ALL_CONFIGS:
            train_model(config, example, bas, use_subset, use_trainval, cpus, gpus)
