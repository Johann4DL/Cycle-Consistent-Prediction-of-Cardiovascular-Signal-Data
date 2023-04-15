import torch
import config
import pandas as pd

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

def save_predictions(loader, gen_A2B, gen_B2A, DEVICE):
    for sig_A, sig_B in loader:
        #convert to float16
        sig_A = sig_A.float()
        sig_B = sig_B.float()
    
        #move to GPU
        sig_A = sig_A.to(DEVICE)
        sig_B = sig_B.to(DEVICE)

        fake_B = gen_A2B(sig_A)
        fake_A = gen_B2A(sig_B)

        #reshape to 1D
        fake_B = fake_B.reshape(-1)
        fake_A = fake_A.reshape(-1)
        sig_A = sig_A.reshape(-1)
        sig_B = sig_B.reshape(-1)

        #calculate mse loss of fake signals and real signals
        #mse_G_A2B = mse(fake_B, sig_B)
        #mse_G_B2A = mse(fake_A, sig_A)
        #print('MSE G_A2B: {:.4f}, MSE G_B2A: {:.4f}'.format(mse_G_A2B.item(), mse_G_B2A.item()))

        #save generated signals as csv in one file
        df = pd.DataFrame({'sig_A': sig_A.cpu().numpy(), 'fake_A': fake_A.cpu().numpy(), 
                        'sig_B': sig_B.cpu().numpy(), 'fake_B': fake_B.cpu().numpy()})
        df.to_csv('generated_signals.csv', index=True)
