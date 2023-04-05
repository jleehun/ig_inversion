
# hard copy from 
# https://github.com/fxnnxc/counterfactuals/blob/3b9470648755b2f534aa8802782e3a65c0fccbd8/counterfactuals/adv.py#L105

import torch 

from tqdm import tqdm 
def run_adv_attack(x,
                   z,
                   optimizer, 
                   classifier,
                   g_model, 
                   target_class: int,
                   attack_style: str,
                   save_at: float,
                   num_steps: int,
                   maximize: bool):
    """
    run optimization process on x or z for num_steps iterations
    early stopping when save_at is reached
    if not return None
    """
    target = torch.LongTensor([target_class]).to(x.device)

    softmax = torch.nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    with tqdm(total=num_steps) as progress_bar:
        for step in range(num_steps):
            optimizer.zero_grad()

            if attack_style == "z":
                x = g_model.decode(z)

            # assert that x is a valid image
            x.data = torch.clip(x.data, min=0.0, max=1.0)

            if "UNet" in type(classifier).__name__:
                _, regression = classifier(x)
                # minimize negative regression to maximize regression
                loss = -regression if maximize else regression

                progress_bar.set_postfix(regression=regression.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                if (maximize and regression.item() > save_at) or (not maximize and regression.item() < save_at):
                    return x

            else:
                prediction = classifier(x)
                acc = softmax(prediction)[torch.arange(0, x.shape[0]), target]
                loss = loss_fn(prediction, target)

                progress_bar.set_postfix(acc_target=acc.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                # early stopping
                if acc > save_at:
                    return x

            loss.backward()
            optimizer.step()

    return None