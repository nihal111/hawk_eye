set -ex
python3 test.py --dataroot ./datasets/football/ --name rgb2edge  --model pix2pix --netG unet_256 --direction AtoB --dataset_mode single --norm batch
