import argparse
import os
import torch
from dataloaders.datasets import DatasetInference
from torchvision.transforms import ToPILImage
from models.Dillated3DUnet import UNet3DLite
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_db_path', type=str, required=True, default=r'/mnt/f/Projects/video_interp_pl/DB/vimeo_septuplet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img-width', type=int, required=True, default=448)
    parser.add_argument('--img-height', type=int, required=True, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # init model
    model = UNet3DLite(init_filters=32).eval().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_dataset = DatasetInference(args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=7, num_workers=1)

    if not os.path.isdir(r'preds'):
        os.makedirs(r'preds')

    img_index = 0
    for img_seq in tqdm(test_loader):
        if img_seq.shape[0] >= 7:
            in_0 = img_seq[0, :, :, :][None, None, :, :, :]
            in_1 = img_seq[2, :, :, :][None, None, :, :, :]
            in_2 = img_seq[4, :, :, :][None, None, :, :, :]
            in_3 = img_seq[6, :, :, :][None, None, :, :, :]
            net_in = torch.cat((in_0, in_1, in_2, in_3), dim=1).to(device)
            net_in = torch.swapaxes(net_in, 1, 2)
            pred = model.forward(net_in)

            pred = torch.clamp(pred, 0.0, 1.0)

            pred_0 = ToPILImage()(pred[0, :, 0, :, :])
            pred_1 = ToPILImage()(pred[0, :, 1, :, :])
            pred_2 = ToPILImage()(pred[0, :, 2, :, :])

            pred_0.save(r'preds/pred_%5.5d.png' % img_index)
            img_index += 1
            pred_1.save(r'preds/pred_%5.5d.png' % img_index)
            img_index += 1
            pred_2.save(r'preds/pred_%5.5d.png' % img_index)
            img_index += 1