import torch, hydra, os, logging
from transformers import set_seed
from portrait_datasets import PortraitDataset
from portraitnet import PortraitNet
from tqdm import tqdm
from trainer import PortraitTrainer

def get_parameters(model, args):
    lr_0 = []
    lr_1 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'deconv' in key:
            lr_0.append(value)
        else:
            lr_1.append(value)
    params = [{'params': [param for key, param in params_dict.items() if 'deconv' in key], 'lr': args.train.learning_rate * 0},
              {'params': [param for key, param in params_dict.items() if 'deconv' not in key], 'lr': args.train.learning_rate * 1}]
    return params, [0., 1.]

@hydra.main(config_path=".", config_name="config.yaml", version_base='1.1')
def main(args):
    # logging
    logging_path = os.path.join(args.output_path, "train.log")
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(
        filename=logging_path,  # Specify the name of the log file
        level=logging.DEBUG,    # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    set_seed(args.train.seed)

    train_dataset = PortraitDataset(args, istrain=True)
    valid_dataset = PortraitDataset(args, istrain=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.workers)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.workers)

    model = PortraitNet(addEdge=True)
    params, multiple = get_parameters(model, args)
    optimizer = torch.optim.Adam(params, args.train.learning_rate, weight_decay=args.train.weight_decay) 

    pretrained_state_dict = torch.load(args.train.pretrained_state_dict)
    model_state_dict = model.state_dict()
    used_param_count, unused_params = 0, []
    for key, param in pretrained_state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape == param.shape:
            model_state_dict[key] = param
            used_param_count += 1
        else:
            unused_params.append(key)
    model.load_state_dict(model_state_dict)
    print(f"Pretrained param: {len(pretrained_state_dict)}, new model param: {len(model_state_dict)}\ninitialized from pretrained: {used_param_count}/{len(model_state_dict)}\nunused pretrained params: {unused_params}")

    model.to(args.device)
    
    trainer = PortraitTrainer(
        args=args, 
        model=model, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader,
        test_dataloader=valid_dataloader,
        multiple=multiple,
    )
    trainer.train()
    
    progress_bar = tqdm(total=args.train.n_epoch * len(dataloader), desc='training...')
    for epoch in range(args.train.n_epoch):
        lr = args.train.learning_rate * (0.95 ** (epoch // 20))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr * multiple[i]

        # train
        

        # test
        model.eval()

        # save_checkpoint()
        progress_bar.update(1)
    
    progress_bar.close()
    return 




if __name__ == '__main__':
    main()