from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from torch import nn
import os
import argparse


def main(args):

    backbone_names = args.backbones.split('+')
    dataset_names = args.datasets.split('+')

    for dataset in dataset_names:
        for backbone in backbone_names:
            print("Working on [DATASET: %s] with [BACKBONE: %s]" %
                  (dataset, backbone))

            # Configure testset path
            test_rgb_path = os.path.join(args.input_root, dataset, 'RGB')
            test_dep_path = os.path.join(args.input_root, dataset, 'depth')

            res_path = os.path.join(args.save_root, 'BiANet_' + backbone,
                                    dataset)
            os.makedirs(res_path, exist_ok=True)
            test_loader = get_loader(test_rgb_path,
                                     test_dep_path,
                                     224,
                                     1,
                                     num_thread=8,
                                     pin=True)

            # Load model and parameters
            exec('from models import BiANet_' + backbone)
            model = eval('BiANet_' + backbone).BiANet()
            pre_dict = torch.load(
                os.path.join(args.param_root, 'BiANet_' + backbone + '.pth'))
            device = torch.device("cuda")
            model.to(device)
            if backbone == 'vgg16':
                model = torch.nn.DataParallel(model, device_ids=[0])
            model.load_state_dict(pre_dict)
            model.eval()

            # Test Go!
            tensor2pil = transforms.ToPILImage()
            with torch.no_grad():
                for batch in test_loader:
                    rgbs = batch[0].to(device)
                    deps = batch[1].to(device)
                    name = batch[2][0]
                    imsize = batch[3]

                    scaled_preds = model(rgbs, deps)

                    res = scaled_preds[-1]

                    res = nn.functional.interpolate(res,
                                                    size=imsize,
                                                    mode='bilinear',
                                                    align_corners=True).cpu()
                    res = res.squeeze(0)
                    res = tensor2pil(res)
                    res.save(os.path.join(res_path, name[:-3] + 'png'))

    print('Outputs were saved at:' + args.save_root)


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--backbones',
                        default='vgg16',
                        type=str,
                        help="Options: 'vgg11','vgg16','res50', 'res2_50")
    parser.add_argument(
        '--datasets',
        default='NJU2K_Test',
        type=str,
        help="Options: 'NJU2K_TEST', 'NLPR_TEST','DES','SSD','STERE','SIP'")
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_root',
                        default='param',
                        type=str,
                        help='folder for pre-trained model')
    parser.add_argument('--input_root',
                        default='./Testset',
                        type=str,
                        help='dataset root')
    args = parser.parse_args()
    parser.add_argument('--save_root',
                        default='./SalMap',
                        type=str,
                        help='Output folder')
    args = parser.parse_args()
    main(args)
