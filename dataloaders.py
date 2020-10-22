from torch.utils.data import DataLoader
from datasets.sketch2color_dataset import Sketch2ColorDataset
from datasets.gray2color_dataset import Gray2ColorDataset


def get_dataloader(args, name):
    if task == 'sketch2color':
        dataset = Sketch2ColorDataset
    elif args.task == 'gray2color':
        dataset = Gray2ColorDataset
    else:
        raise NotImplementedError()

    if name == 'train':
        train_dataset = dataset(args, 'train')
        return DataLoader(
            dataset=train_dataset,
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=True
        )
    elif name == 'val': 
        val_dataset = dataset(args, 'val')
        return DataLoader(
            dataset=val_dataset,
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=False
        )
    elif name == 'test':
        test_dataset = dataset(args, 'test')
        return DataLoader(
            dataset=test_dataset,
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        raise NotImplementedError()