import torch
import config
import pytorch_lightning as pl
from models import wGANGP

if __name__=="__main__":
    args = config.get_config()
    print(args)

    model = wGANGP(args)
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.epochs)
    trainer.fit(model)
