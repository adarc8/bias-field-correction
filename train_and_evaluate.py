import torch
import config
import numpy as np
from tqdm import tqdm
from piq import psnr, ssim
from utils import accuracy_by_disc, get_uc_loss, get_clean_division


def train_or_evaluate(training_flag, epoch, loader, L1, L2, BCE, device, opt_gen1, opt_gen2, opt_disc, gen1, gen2, disc,
                      stn, opt_stn, g1_scaler, g2_scaler, d_scaler):

    rec_loss_vec, gen1_loss_vec, gen2_loss_vec, ADV_loss_vec, disc_loss_vec = [], [], [], [], []
    bias_psnr, clean_psnr, bias_ssim, clean_ssim, accuracy_real_vec, accuracy_fake_vec = [], [], [], [], [], []
    dirty_psnr, dirty_ssim, flip_loss_vec, soft_loss_vec, uc_loss_vec, division_clean_psnr, division_clean_ssim = [], [], [], [], [], [], []

    loop = tqdm(loader, leave=True)
    itreation = 0

    lam_adv = config.LAMBDA_ADV
    lam_sym = config.LAMBDA_SYM
    lam_uc = config.LAMBDA_UC

    for idx, (input_img, clean_ref, bias_ref, seg) in enumerate(loop):

        input_img = input_img.to(device)
        clean_ref = clean_ref.to(device)
        bias_ref = bias_ref.to(device)
        # seg = seg.to(device)

        # ~~~~~~~~~~~~~~~~~~ Part (b) # Training Bias Disc ~~~~~~~~~~~~~~~~~~

        with torch.set_grad_enabled(training_flag):
            with torch.cuda.amp.autocast():
                est_clean = gen1(input_img)
                est_bias = gen2(input_img)
                est_dirty = est_clean * est_bias
                est_clean_by_division = get_clean_division(input_img, est_bias)

                bias_of_est_clean = gen2(est_clean)

                # STN PART
                mirror = torch.flip(est_clean, [3])
                concat = torch.cat([est_clean, mirror], 1)
                _, est_mirror = stn(concat)

                disc_fake = disc(est_bias.detach())
                disc_real = disc(bias_ref)

                disc_fake_loss = BCE(disc_fake, torch.zeros_like(disc_fake))
                disc_real_loss = BCE(disc_real, torch.ones_like(disc_real))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2

        if training_flag:
            opt_disc.zero_grad()
            d_scaler.scale(disc_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
                # disc_loss.backward()
                # opt_disc.step()

        accuracy_real, accuracy_fake = accuracy_by_disc(disc_real, disc_fake, training_flag, 1000 + epoch * 1000,
                                                        itreation, create_video=training_flag)
        itreation += 1

        # Part (a)

        with torch.cuda.amp.autocast():
            # Reconstruction Loss
            REC_loss = L2(input_img, est_dirty)  # + L1(input_img,est_dirty)
            # Adversarial Loss
            disc_fake_hat = disc(est_bias)
            ADV_loss = BCE(disc_fake_hat, torch.ones_like(disc_fake_hat))
            # Symmetry Loss
            symmetry_loss = L1(mirror, est_mirror)  # + (1 - weight_flip_l1) * L2(est_clean,fliped_est_clean)
            # uc Loss
            uc_loss = get_uc_loss(input_img,bias_of_est_clean,L2)

            gen1_loss = REC_loss + lam_sym *symmetry_loss  + lam_uc * uc_loss

            gen2_loss = REC_loss + lam_adv * ADV_loss

        if training_flag:
            opt_stn.zero_grad()
            symmetry_loss.backward(retain_graph=True)  # retain_graph=True?
            opt_stn.step()
            opt_gen1.zero_grad()
            g1_scaler.scale(gen1_loss).backward(retain_graph=True)
            # gen1_loss.backward(retain_graph=True)
            opt_gen2.zero_grad()
            g2_scaler.scale(gen2_loss).backward()
            g2_scaler.step(opt_gen2)
            g2_scaler.update()
            # gen2_loss.backward()
            # opt_gen2.step()

            g1_scaler.step(opt_gen1)
            g1_scaler.update()
            # opt_gen1.step()


        est_clean = est_clean * (input_img > 0.01)

        try:
            dirty_psnr.append(psnr(input_img, est_dirty).item())
            dirty_ssim.append(ssim(input_img, est_dirty).item())

            bias_psnr.append(psnr(bias_ref, est_bias).item())
            clean_psnr.append(psnr(est_clean, clean_ref).item())

            division_clean_psnr.append(psnr(est_clean_by_division, clean_ref).item())
            division_clean_ssim.append(ssim(est_clean_by_division, clean_ref).item())

            bias_ssim.append(ssim(bias_ref, est_bias).item())
            clean_ssim.append(ssim(est_clean, clean_ref).item())
        except:
            break_point = 1  # explore est dirty and how come it is not in range [0,1]
            print('\nskiping psnr and ssim calculation at epoch: ' + str(epoch))

        gen1_loss_vec.append(gen1_loss.item())
        gen2_loss_vec.append(gen2_loss.item())

        rec_loss_vec.append(REC_loss.item())
        ADV_loss_vec.append(ADV_loss.item())
        disc_loss_vec.append(disc_loss.item())

        accuracy_real_vec.append(accuracy_real.cpu())
        accuracy_fake_vec.append(accuracy_fake.cpu())

        flip_loss_vec.append(symmetry_loss.item())
        soft_loss_vec.append(0)
        uc_loss_vec.append(uc_loss.item())

        #

        if training_flag:
            train_or_val = 'Training: '
        else:
            train_or_val = 'Validation: '

        loop.set_description(train_or_val + '%d/%d' % (epoch, config.NUM_EPOCHS))

    return [
        np.mean(rec_loss_vec), np.mean(ADV_loss_vec), np.mean(disc_loss_vec),  # 0 1 2
        np.mean(gen1_loss_vec), np.mean(gen2_loss_vec),  # 3 4
        np.mean(bias_psnr), np.mean(clean_psnr), np.mean(bias_ssim), np.mean(clean_ssim),  # 5 6 7 8
        np.mean(accuracy_real_vec), np.mean(accuracy_fake_vec), np.mean(dirty_psnr), np.mean(dirty_ssim),
        np.mean(flip_loss_vec), np.mean(soft_loss_vec), np.mean(uc_loss_vec),
        np.mean(division_clean_psnr), np.mean(division_clean_ssim)]  # 13 14 15 16


def Write_On_TB(data, writer, step):
    writer.add_scalar('REC Loss', data[0], global_step=step)
    writer.add_scalar('ADV Loss', data[1], global_step=step)
    writer.add_scalar('D Loss', data[2], global_step=step)

    writer.add_scalar('Gen1 Loss', data[3], global_step=step)
    writer.add_scalar('Gen2 Loss', data[4], global_step=step)

    writer.add_scalar('Bias PSNR', data[5], global_step=step)
    writer.add_scalar('Clean PSNR', data[6], global_step=step)
    writer.add_scalar('Bias SSIM', data[7], global_step=step)
    writer.add_scalar('Clean SSIM', data[8], global_step=step)

    writer.add_scalar('Accuracy Disc (Real)', data[9], global_step=step)
    writer.add_scalar('Accuracy Disc (Fake)', data[10], global_step=step)

    writer.add_scalar('REC PSNR', data[11], global_step=step)
    writer.add_scalar('REC SSIM', data[12], global_step=step)

    writer.add_scalar('Flip Loss', data[13], global_step=step)
    writer.add_scalar('Soft Loss', data[14], global_step=step)
    writer.add_scalar('uc Loss', data[15], global_step=step)

    writer.add_scalar('Clean PSNR by division', data[16], global_step=step)
    writer.add_scalar('Clean SSIM by division', data[17], global_step=step)
