import torch
from utils import save_some_examples, clean_folder, get_input_from_user, seed_everything
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MyDataset
from unet_model import UNet
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_and_evaluate import train_or_evaluate, Write_On_TB
from stn_model import STN


seed_everything()


def main():
    if float(get_input_from_user("Clean TensorBoard folder? 0 or 1")):
        clean_folder("TensorBoard")
    device = "cuda:" + str(get_input_from_user("Choose number GPU to use (0,1,2,3):"))

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    BCE = nn.BCEWithLogitsLoss()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if config.IBSR_or_iSeg == 'iSeg':
        folds = 5
    elif config.IBSR_or_iSeg == 'IBSR':
        folds = 6

    for fold in range(folds):  # Only fold0 test right now

        train_dataset = MyDataset(config.FOLDS_PATHS, fold_amounts=folds, valid_fold=fold, validation_flag=False)
        val_dataset = MyDataset(config.FOLDS_PATHS, fold_amounts=folds, valid_fold=fold, validation_flag=True)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        print(f'Fold Number: {fold}')

        clean_PSNR_array, bias_PSNR_array, clean_SSIM_array = [], [], []
        best_psnr = 0

        gen1 = UNet(1, 1).to(device)
        gen2 = UNet(1, 1).to(device)

        disc = Discriminator(in_channels=1).to(device)

        stn = STN(1).to(device)

        opt_gen1 = optim.Adam(gen1.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
        opt_gen2 = optim.Adam(gen2.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
        g1_scaler = torch.cuda.amp.GradScaler()
        g2_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
        opt_stn = optim.Adam(stn.parameters(), lr=1e-4, weight_decay=0)

        description = f'Run'

        train_writer = SummaryWriter(f'TensorBoard/Train_{description}')
        valid_writer = SummaryWriter(f'TensorBoard/Valid_{description}')

        for epoch in range(1, config.NUM_EPOCHS):

            gen1.train()
            gen2.train()
            disc.train()
            stn.train()
            training_flag = True

            data = train_or_evaluate(training_flag, epoch, train_loader, L1, L2, BCE, device, opt_gen1, opt_gen2, opt_disc,
                                     gen1, gen2, disc, stn, opt_stn, g1_scaler, g2_scaler, d_scaler)

            Write_On_TB(data, train_writer, epoch)

            gen1.eval()
            gen2.eval()
            disc.eval()
            stn.eval()
            training_flag = False
            data = train_or_evaluate(training_flag, epoch, val_loader, L1, L2, BCE, device, opt_gen1, opt_gen2, opt_disc,
                                     gen1, gen2, disc, stn, opt_stn, g1_scaler, g2_scaler, d_scaler)

            Write_On_TB(data, valid_writer, epoch)

            clean_PSNR_array.append(data[6])
            bias_PSNR_array.append(data[5])
            clean_SSIM_array.append(data[8])

            if data[5] > best_psnr:
                best_psnr = data[5]

            save_some_examples(gen1, gen2, stn, train_loader, epoch,
                               folder=f"samples_druing_training/{description}",
                               img_name=f"epoch={epoch + 1000}_train.png", device=device)

            save_some_examples(gen1, gen2, stn, val_loader, epoch, folder=f"samples_druing_training/{description}",
                               img_name=f"epoch={epoch + 1000}_val.png", device=device, create_video=True)



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    main()
