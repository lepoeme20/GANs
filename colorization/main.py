import config, trainer

if __name__ == "__main__":
    args = config.get_config()
    print(args)

    print("Main algorithm is training")
    trainer.train(args)