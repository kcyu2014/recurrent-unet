import logging 
from ptsemseg.schedulers.schedulers import *

logger = logging.getLogger('ptsemseg')

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'multi_step': MultiStepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR,
                 'StepLR': StepLR}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)
    
    s_type = scheduler_dict['name']
    scheduler_dict.pop('name')

    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            scheduler_dict))

    warmup_dict = {} 
    if 'warmup_iters' in scheduler_dict:
        # This can be done in a more pythonic way... 
        warmup_dict['warmup_iters'] = scheduler_dict.get('warmup_iters', 100)
        warmup_dict['mode'] = scheduler_dict.get('warmup_mode', 'linear')
        warmup_dict['gamma'] = scheduler_dict.get('warmup_factor', 0.2)

        logger.info('Using Warmup with {} iters {} gamma and {} mode'.format(
                                        warmup_dict['warmup_iters'],
                                        warmup_dict['gamma'],
                                        warmup_dict['mode']))

        scheduler_dict.pop('warmup_iters', None) 
        scheduler_dict.pop('warmup_mode', None)
        scheduler_dict.pop('warmup_factor', None) 

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    StepLR_dict = {}
    if 'lr_decay_step_size' in scheduler_dict:
        # This can be done in a more pythonic way...
        gamma = scheduler_dict.get('lr_decay_factor_gamma', 0.1)
        step_size = scheduler_dict.get('lr_decay_step_size', 10000)
        logger.info('Using StepLR with {} step_iters and {} gamma lr_decay'.format(
            gamma, step_size))

        scheduler_dict.pop('lr_decay_step_size', None)
        scheduler_dict.pop('lr_decay_factor_gamma', None)

        scheduler_dict['step_size'] = step_size
        scheduler_dict['gamma'] = gamma

        steplr_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return steplr_scheduler

    return key2scheduler[s_type](optimizer, **scheduler_dict)
