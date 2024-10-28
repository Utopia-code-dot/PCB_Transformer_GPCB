from sample import  sample

if __name__ == "__main__":
    mode = "sample"  # if train, training from beginning, if sample, loading model from checkpoint and run sampling

    if mode == "train":
        # training loop
        from train import train
        out_dir = train()
        # sampling loop
        sample(out_dir)

    if mode == "sample":
        # sampling loop
        sample('./out_train/model_20240719_2036')