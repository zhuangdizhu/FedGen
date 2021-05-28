from torch.optim import Optimizer

class pFedIBOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr)
        super(pFedIBOptimizer, self).__init__(params, defaults)

    def step(self, apply=True, lr=None, allow_unused=False):
        grads = []
        # apply gradient to model.parameters, and return the gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                grads.append(p.grad.data)
                if apply:
                    if lr == None:
                        p.data= p.data - group['lr'] * p.grad.data
                    else:
                        p.data=p.data - lr * p.grad.data
        return grads


    def apply_grads(self, grads, beta=None, allow_unused=False):
        #apply gradient to model.parameters
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                p.data= p.data - group['lr'] * grads[i] if beta == None else p.data - beta * grads[i]
                i += 1
        return


class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu)
        super(FedProxOptimizer, self).__init__(params, defaults)

    def step(self, vstar, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, pstar in zip(group['params'], vstar):
                # w <=== w - lr * ( w'  + lambda * (w - w* ) + mu * w )
                p.data=p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - pstar.data.clone()) + group['mu'] * p.data)
        return group['params'], loss
