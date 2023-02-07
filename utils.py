import torch
import config

def save_checkpoint(model, optimizer, path="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint at location: ", path)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint from location: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
