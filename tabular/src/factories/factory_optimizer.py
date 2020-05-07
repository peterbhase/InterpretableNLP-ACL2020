"""creates various optimizers"""
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class OptimizerFactory():
    """
    OptimizerFactory contains wrappers to create various optimizers.
    Assumes using adam for MLP and sgd for proto model
    """
    @staticmethod
    def create(args, tagger):
        if args.opt == 'sgd':
            # don't weight decay on prototypes
            if hasattr(tagger, 'prototypes_shape'):
                specs = [
                        {'params': params, 'lr': args.lr, 'weight_decay': args.weight_decay} \
                        if params.shape != tagger.prototypes_shape \
                        else {'params': params, 'lr': args.lr}
                        for params in tagger.parameters() 
                        ]
            else:
                specs = [{'params': params, 'lr': args.lr, 'weight_decay': args.weight_decay} for params in tagger.parameters()]
            optimizer = optim.SGD(specs, momentum=args.momentum)
        elif args.opt == 'adam':
            optimizer = optim.Adam(tagger.parameters(), weight_decay = args.weight_decay, lr=args.lr, betas=(0.9, 0.999))
        else:
            raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch))
        return optimizer, scheduler
