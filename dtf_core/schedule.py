import torch
import torch.nn as nn
import torch.optim as optim
import math

class _LRSchedulerBase():
    def __init__( self, optimizer ):
        self._optimizer = optimizer

    def _optim_step( self ):
        self._optimizer.step()

    def _apply_optim_lr( self, lr ):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def zero_grad(self):
        """
        Zero out the gradients with the inner optimizer
        """
        self._optimizer.zero_grad()

    def get_last_lr( self ):
        """
        Get last applied learning rate
        """
        for param_group in self._optimizer.param_groups:
            return param_group['lr']

class ConstantLR(_LRSchedulerBase):
    def __init__(
            self,
            optimizer,
            lr ):
        super(ConstantLR,self).__init__(optimizer)
        self.lr = lr

    def step_lr(self,steps=1):
        """
        Step with the inner optimizer
        """

        self._apply_optim_lr(self.lr)
        #self._optim_step()

class LinearAnnealing(_LRSchedulerBase):
    def __init__(
            self,
            optimizer,
            nominal_lr,
            warmup_steps,
            constant_steps,
            total_steps,
            min_lr=1e-7 ):
        super(LinearAnnealing,self).__init__(optimizer)
        self.steps = 0
        self.nominal_lr = nominal_lr
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def _get_lr(self):
        if self.steps < self.warmup_steps:
            return max( self.min_lr, self.nominal_lr * self.steps/self.warmup_steps )
        elif self.steps < self.warmup_steps + self.constant_steps:
            return self.nominal_lr
        else:
            n = self.steps - self.warmup_steps - self.constant_steps
            t = self.total_steps - self.warmup_steps - self.constant_steps
            if t > 0:
                #return max( self.min_lr, self.nominal_lr * math.exp(-3*n/t) )
                return max( self.min_lr, self.nominal_lr * (1-n/t) )
            else:
                return self.nominal_lr

    def step_lr(self,steps=1):
        """
        Step with the inner optimizer
        """

        self.steps += steps
        lr = self._get_lr()

        self._apply_optim_lr(lr)
        #self._optim_step()

class CosineAnnealing(_LRSchedulerBase):
    def __init__(
            self,
            optimizer,
            nominal_lr: float,
            nominal_period: float,
            warmup_steps: int = 0,
            lr_decreasing: float = 1.,
            period_dilation: float = 1.,
            min_lr: float = 1e-7 ):
        super(CosineAnnealing,self).__init__(optimizer)

        self.steps = 0
        self.nominal_lr = nominal_lr
        self.nominal_period = nominal_period
        self.warmup_steps = warmup_steps
        self.lr_decreasing = lr_decreasing
        self.period_dilation = period_dilation
        self.min_lr = min_lr

    def _get_lr(self):
        dilated = (self.steps/self.nominal_period)**self.period_dilation
        cycle = math.floor(dilated)
        max_lr = self.lr_decreasing**cycle
        cycle_begin = cycle**(1./self.period_dilation) * self.nominal_period
        cycle_end = (cycle+1)**(1./self.period_dilation) * self.nominal_period
        factor = self.nominal_lr * self.lr_decreasing**cycle

        cycle_step = self.steps - cycle_begin
        if cycle_step < self.warmup_steps:
            lr = cycle_step/self.warmup_steps
        else:
            lr = 0.5 + 0.5*math.cos( math.pi * (cycle_step-self.warmup_steps)/(cycle_end-cycle_begin-self.warmup_steps) )
        return self.min_lr + (factor-self.min_lr)*lr

    def step_lr(self,steps=1):
        """
        Step with the inner optimizer
        """

        self.steps += steps
        lr = self._get_lr()

        self._apply_optim_lr(lr)
        #self._optim_step()

class OneCycleLR(_LRSchedulerBase):
    def __init__( self,
            optimizer,
            max_lr,
            total_steps,
            batch_size,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95 ):
        super(OneCycleLR,self).__init__(optimizer)
        self.one_cycle = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps//batch_size,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum)
        self.batch_size = batch_size

    def step_lr(self,steps=1):
        """
        Step with the inner optimizer
        """
        self.one_cycle.step()
