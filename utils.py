import torch
import config
from torchvision.utils import save_image
import os, shutil
import numpy as np
# import moviepy.video.io.ImageSequenceClip
import random
import pylab as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from scipy import ndimage


def normalize_imgs(imgs_tensor):  # imgs_tensor expected to be [BS,1,x,y] example [32,1,256,256]
    minimum = imgs_tensor.amin(dim=(1, 2, 3), keepdim=True)
    maximum = imgs_tensor.amax(dim=(1, 2, 3), keepdim=True)
    imgs_tensor = (imgs_tensor - minimum) / (torch.max((maximum - minimum), 1e-5 * torch.rand_like(maximum)))

    return imgs_tensor


def save_some_examples(gen1, gen2, stn, loader, epoch, folder, img_name, device, create_video=False):
    input_img, clean_ref, bias_ref, seg = next(iter(loader))
    input_img = input_img[:7, :, :, :].to(device)
    clean_ref = clean_ref[:7, :, :, :].to(device)
    bias_ref = bias_ref[:7, :, :, :].to(device)

    with torch.no_grad():
        est_clean = gen1(input_img)
        est_clean = est_clean * (input_img > 0.01)
        est_bias = gen2(input_img)

        mirror = torch.flip(est_clean, [3])
        concat = torch.cat([est_clean, mirror], 1)
        _, est_mirror = stn(concat)

        clean_division = get_clean_division(input_img, est_bias)

    x_col = torch.cat([input_img, est_clean, mirror, est_mirror, clean_division, clean_ref, est_bias, bias_ref], -2)
    texts = ['input_img', 'est_clean', 'mirror', 'est_mirror', 'clean_divison', 'clean_ref', 'est_bias', 'bias_ref']
    x_col = add_text(x_col, texts, res=input_img.shape[-1])

    dst_path = os.path.join(folder, img_name)
    new_path, filename = os.path.split(dst_path)
    if not os.path.exists(new_path):  # if folder doesnt exist
        os.makedirs(new_path)  # create it
    save_image(x_col, dst_path)

    # if epoch%50 == 0 and epoch!= 0 and create_video :
    #     video_of_imgs(epoch,folder)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def write_gen_weights(writer, encoder, decoder, step):
    # down/ up is a block that have conv sequential and droput.
    # conv sequential has conv and relu,batchnorm, indexed 0 1 2. So basicly I take
    # each block, then the sequential of the block, and then the first elemnt of seq which is conv
    # initial_down,bottle, and final_up are not a block, its just squential
    writer.add_histogram('down1conv1', encoder.down1.conv1.conv[0].weight, global_step=step)
    writer.add_histogram('down1conv2', encoder.down1.conv2.conv[0].weight, global_step=step)
    writer.add_histogram('down2conv1', encoder.down2.conv1.conv[0].weight, global_step=step)
    writer.add_histogram('down2conv2', encoder.down2.conv2.conv[0].weight, global_step=step)
    writer.add_histogram('mu_conv', encoder.mu_conv.weight, global_step=step)
    writer.add_histogram('logvar_conv', encoder.logvar_conv.weight, global_step=step)
    writer.add_histogram('up1conv1', decoder.up1.conv1.conv[0].weight, global_step=step)
    writer.add_histogram('up1conv2', decoder.up1.conv2.conv[0].weight, global_step=step)
    writer.add_histogram('up2conv1', decoder.up2.conv1.conv[0].weight, global_step=step)
    writer.add_histogram('up2conv2', decoder.up2.conv2.conv[0].weight, global_step=step)


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_input_from_user(question):
    import tkinter as tk
    from tkinter import simpledialog
    # import winsound
    frequency = 200  # Set Frequency To 2500 Hertz
    duration = 200  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)
    # winsound.Beep(frequency, duration)

    ROOT = tk.Tk()

    ROOT.withdraw()
    # the input dialog
    print('\nSearch for pop-up window\n')
    return simpledialog.askstring(title="Pop-up Window", prompt=question)


def video_of_imgs(epoch, folder, ends_with="val.png", fps=10):
    image_files = [os.path.join(folder, img)
                   for img in sorted(os.listdir(folder))
                   if img.endswith(ends_with)]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    video_name = os.path.join(folder, 'Training-Process.mp4')
    clip.write_videofile(video_name)

    project_name = os.getcwd().split('/')[-1]
    mail_title = project_name + ' Process Video'
    mail_content = 'From epoch: ' + str(epoch)
    SendMail(mail_title, mail_content, video_name)


def model_system(training_flag, gen1, gen2, input_img):
    with torch.set_grad_enabled(training_flag):
        est_clean = (gen1(input_img))
        est_bias = (gen2(input_img))
        est_dirty = est_clean * est_bias
    return est_clean, est_bias, est_dirty


def accuracy_by_disc(pred_real, pred_fake, training_flag, epoch, itr, create_video):
    accuracy_real = (pred_real > 0.5).sum() / (pred_real == pred_real).sum()
    accuracy_fake = (pred_fake < 0.5).sum() / (pred_fake == pred_fake).sum()

    # if itr==0:
    # # normalize values:
    #     max_pred = (torch.stack([pred_real,pred_fake])).max().cpu().detach() # size of stack: torch.Size([2, 32, 1, 12, 12])
    #     min_pred = (torch.stack([pred_real,pred_fake])).min().cpu().detach()

    #     pred_real = torch.flatten(pred_real).cpu().detach()
    #     pred_fake = torch.flatten(pred_fake).cpu().detach()
    #     # avg_mean = avg_mean.cpu().detach()
    #     # pred_real = (pred_real - min_pred)/(max_pred-min_pred)
    #     # pred_fake = (pred_fake - min_pred)/(max_pred-min_pred)
    #     # avg_mean = (pred_real.mean()+pred_fake.mean())/2

    #     #                                                                        variance=0.1
    #     plt.scatter(pred_real,torch.ones_like(pred_fake)+0.01+torch.randn_like(pred_real)*np.sqrt(0.00001),alpha=0.7,s=10,c='blue')
    #     plt.scatter(pred_fake,torch.ones_like(pred_fake)-0.01+torch.randn_like(pred_real)*np.sqrt(0.00001),alpha=0.7,s=10,c='red')
    #     plt.axvline(x=0.5,c='black')
    #     plt.ylim(1.05,0.95)
    #     plt.xlim(0,1)
    #     plt.grid(True)
    #     leg = plt.legend(('Th','Real','Fake'),loc="upper left")
    #     for lh in leg.legendHandles:
    #         lh.set_alpha(1)

    #     if training_flag:
    #         word = 'train'
    #     else:
    #         word = 'val'

    #     plt.savefig(f'disc desicion/{epoch+itr}_{word}.png')
    #     plt.clf() # this clean the figure. if u want the figure for each epoch than # this command
    #     if epoch/1000 % 50 == 0 and create_video:
    #         video_of_imgs(epoch,'disc desicion',ends_with=f"{word}.png",fps=10)

    return accuracy_real, accuracy_fake


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_uc_loss(input_img, bias_of_est_clean, loss_critertion):
    mask = input_img > 0
    bias_mean = (bias_of_est_clean * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    bias_mean = bias_mean.view(bias_mean.shape[0], 1, 1, 1)
    soft_loss = loss_critertion(bias_of_est_clean, bias_mean * mask)
    return soft_loss


def denormalize(img, mean=config.MEAN, std=config.STD):
    return img * std + mean


def get_clean_division(input_img, est_bias):
    est_bias = est_bias * (input_img > 0.01)
    est_bias[est_bias < 0.01] = 0.01
    est_bias[est_bias == 0] = 999
    clean_division = input_img / est_bias
    clean_division = clean_division * (input_img > 0.01)
    clean_division = normalize_imgs(clean_division)

    clean_division[clean_division != clean_division] = 0
    clean_division[clean_division == np.inf] = 0

    return clean_division


def add_text(x_col, texts, res=184):
    for idx, text in enumerate(texts):
        img = Image.new('L', (res, res), color=(0))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", res // 6)
        d.text((5, res // 2), text, fill=(255), font=font)
        img = transforms.ToTensor()(np.float32(img) / 255)
        if idx == 0:
            column = img
        else:
            column = torch.cat([column, img], -2)
    device = x_col.device.type + ':' + str(x_col.device.index)  # get device
    column = torch.unsqueeze(column, 0).to(device).to(device)

    x_col = torch.cat([column, x_col], 0)
    return x_col


def normalize_to(imgs_tensor, b=1):  # imgs_tensor expected to be [BS,1,x,y] example [32,1,256,256]
    minimum = imgs_tensor.amin(dim=(1, 2, 3), keepdim=True)
    maximum = imgs_tensor.amax(dim=(1, 2, 3), keepdim=True)
    # epsilon = 1e-8
    imgs_tensor = b * (imgs_tensor - minimum) / (torch.max((maximum - minimum), 1e-5 * torch.rand_like(maximum)))

    # imgs_tensor = imgs_tensor / imgs_tensor.amax(dim=(1,2,3), keepdim=True)
    # imgs_tensor[imgs_tensor<0] = 0
    return imgs_tensor


def histogram_as_matrix_img(img, dont_include_zeros=True, gray_scale=True):
    if dont_include_zeros:
        plt.hist(img[img != 0].ravel(), bins=256, range=(0, 1), fc='k', ec='k')  # calculating histogram()
    else:
        plt.hist(img.ravel(), bins=256, range=(0, 1), fc='k', ec='k')  # calculating histogram()
    plt.show()
    return


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded) # print the matrix
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


