import torch
import torchvision
import torchvision.transforms as transforms
from utils import *


def train(args):

    root = "/media/lepoeme20/Data/daewoo/imgs4GAN/lngc2/"
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.n_cpu)
    # Loss Functions
    criterion = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()


    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)


    # Initialize generator and discriminator
    netG, netD = network_initialization(args)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    batches_done = 0
    prev_g_cost = 1000
    one = torch.tensor(1, dtype=torch.float).to(args.device)
    mone = one * -1

    # imgs, _ = next(iter(train_dataloader))
    for epoch in range(0, args.n_epochs):
        for step, (imgs, _) in enumerate(train_dataloader, 0):
            #######################################################
            # (0) Prepare training data
            ######################################################
            real_Gray = [transforms.Grayscale()(x) for x in imgs]
            real_RGB = imgs.to(args.device)


            #######################################################
            # (1) Update D network
            ######################################################
            for p in netD.parameters():
                p.requires_grad = True

            optimizer_D.zero_grad()

            # Real images
            real_validity = netD(real_imgs)
            d_loss_real = real_validity.mean()
            d_loss_real.backward(mone)

            # Generate a batch of images
            fake_imgs = netG(z)

            # Fake images
            fake_validity = netD(fake_imgs)
            d_loss_fake = fake_validity.mean()
            d_loss_fake.backward(one)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(args, netD, real_imgs.data, fake_imgs.data)
            gradient_penalty.backward()

            # Adversarial loss
            d_loss = d_loss_fake - d_loss_real + gradient_penalty

            optimizer_D.step()