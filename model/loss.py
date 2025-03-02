import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class CoxPHLoss(nn.Module):
    '''
    Loss for CoxPH model, i.e., negative partial log-likelihood loss.
    
    - First, sort data by descending duration
    - Then calculate the loss.

    log_h is the log hazard ratio, i.e., log(h_i) for each sample i.
    
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    
    Reference: 
    - https://github.com/havakv/pycox/blob/master/pycox/models/loss.py#L407
    - https://arxiv.org/abs/1907.00825
    - https://github.com/thbuerg/MetabolomicsCommonDiseases/blob/main/metabolomicstatemodel/source/losses.py
    '''
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, log_h, y, e, eps=1e-7):
        """
        Parameters:
        - log_h: torch.Tensor, shape (batch_size,)
            The log hazard ratio.
        - y: torch.Tensor, shape (batch_size,)
            The observed time of events.
        - e: torch.Tensor, shape (batch_size,)
            The event indicator.
        """
        # Sort
        idx = y.sort(descending=True)[1]
        e = e[idx]
        log_h = log_h[idx]
        
        # Calculate loss
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        if e.sum() > 0:
            loss = - log_h.sub(log_cumsum_h).mul(e).sum().div(e.sum())
        else:
            loss = - log_h.sub(log_cumsum_h).mul(e).sum() # Equal to zero
        
        return loss
      

class MultiTaskLoss(nn.Module):
  '''
  Uncertainty Weighting Multi-task Learning
  
  Ref:
    - https://arxiv.org/abs/1705.07115
    - https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
  '''
  def __init__(self, is_regression):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks, requires_grad=True))

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    return multi_task_losses


class AutomaticWeightedLoss(nn.Module):
    '''
    Automatically weighted multi-task loss.

    Params:
        n: int
            The number of loss functions to combine.
        reduction: str, optional (default='sum')
            Specifies the reduction to apply to the output: 'sum' or 'mean'.
        x: tuple
            A tuple containing multiple task losses.
            
  Ref:
    - https://arxiv.org/abs/1705.07115
    - https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
    - https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
    '''
    def __init__(self, n=2, reduction='sum'):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(n, requires_grad=True)
        self.params = nn.Parameter(params)
        self.reduction = reduction
        
    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            regularization = torch.log(1 + self.params[i] ** 2)
            loss_sum += weighted_loss + regularization

        if self.reduction == 'mean':
            return loss_sum / len(losses)
        return loss_sum
    

class TorchSurvCoxLoss(nn.Module):
    def __init__(self, ties_method: str = "efron", reduction: str = "mean", checks: bool = True):
        super().__init__()
        self.ties_method = ties_method
        self.reduction = reduction
        self.checks = checks

    def forward(self, log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.neg_partial_log_likelihood(log_hz, event, time)

    def neg_partial_log_likelihood(
        self,
        log_hz: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:

        if self.checks:
            self._check_inputs(log_hz, event, time)

        if any([event.sum() == 0, len(log_hz.size()) == 0]):
            warnings.warn("No events OR single sample. Returning zero loss for the batch")
            return torch.tensor(0.0, requires_grad=True)

        # sort data by time-to-event or censoring
        time_sorted, idx = torch.sort(time)
        log_hz_sorted = log_hz[idx]
        event_sorted = event[idx]
        time_unique = torch.unique(time_sorted)

        if len(time_unique) == len(time_sorted):
            # if not ties, use traditional cox partial likelihood
            pll = self._partial_likelihood_cox(log_hz_sorted, event_sorted)
        else:
            # if ties, use either efron or breslow approximation of partial likelihood
            if self.ties_method == "efron":
                pll = self._partial_likelihood_efron(
                    log_hz_sorted,
                    event_sorted,
                    time_sorted,
                    time_unique,
                )
            elif self.ties_method == "breslow":
                pll = self._partial_likelihood_breslow(log_hz_sorted, event_sorted, time_sorted)
            else:
                raise ValueError(
                    f'Ties method {self.ties_method} should be one of ["efron", "breslow"]'
                )

        # Negative partial log likelihood
        pll = torch.neg(pll)
        if self.reduction.lower() == "mean":
            loss = pll.nanmean()
        elif self.reduction.lower() == "sum":
            loss = pll.sum()
        else:
            raise ValueError(
                f"Reduction {self.reduction} is not implemented yet, should be one of ['mean', 'sum']."
            )
        return loss

    def _partial_likelihood_cox(
        self,
        log_hz_sorted: torch.Tensor,
        event_sorted: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the partial log likelihood for the Cox proportional hazards model
        in the absence of ties in event time.
        """
        log_denominator = torch.logcumsumexp(log_hz_sorted.flip(0), dim=0).flip(0)
        return (log_hz_sorted - log_denominator)[event_sorted]

    def _partial_likelihood_efron(
        self,
        log_hz_sorted: torch.Tensor,
        event_sorted: torch.Tensor,
        time_sorted: torch.Tensor,
        time_unique: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the partial log likelihood for the Cox proportional hazards model
        using Efron's method to handle ties in event time.
        """
        J = len(time_unique)

        H = [
            torch.where((time_sorted == time_unique[j]) & (event_sorted == 1))[0]
            for j in range(J)
        ]
        R = [torch.where(time_sorted >= time_unique[j])[0] for j in range(J)]

        m = torch.tensor([len(h) for h in H])
        include = torch.tensor([len(h) > 0 for h in H])

        log_nominator = torch.stack([torch.sum(log_hz_sorted[h]) for h in H])

        denominator_naive = torch.stack([torch.sum(torch.exp(log_hz_sorted[r])) for r in R])
        denominator_ties = torch.stack([torch.sum(torch.exp(log_hz_sorted[h])) for h in H])

        log_denominator_efron = torch.zeros(J).to(log_hz_sorted.device)
        for j in range(J):
            for l in range(1, m[j] + 1):
                log_denominator_efron[j] += torch.log(
                    denominator_naive[j] - (l - 1) / m[j] * denominator_ties[j]
                )
        return (log_nominator - log_denominator_efron)[include]

    def _check_inputs(self, log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
        if not isinstance(log_hz, torch.Tensor):
            raise TypeError("Input 'log_hz' must be a tensor.")

        if not isinstance(event, torch.Tensor):
            raise TypeError("Input 'event' must be a tensor.")

        if not isinstance(time, torch.Tensor):
            raise TypeError("Input 'time' must be a tensor.")

        if len(log_hz) != len(event):
            raise ValueError(
                "Length mismatch: 'log_hz' and 'event' must have the same length."
            )

        if len(time) != len(event):
            raise ValueError(
                "Length mismatch: 'time' must have the same length as 'event'."
            )

        if any(val < 0 for val in time):
            raise ValueError("Invalid values: All elements in 'time' must be non-negative.")

        if any(val not in [True, False, 0, 1] for val in event):
            raise ValueError(
                "Invalid values: 'event' must contain only boolean values (True/False or 1/0)"
            )

    def _partial_likelihood_breslow(
        self,
        log_hz_sorted: torch.Tensor,
        event_sorted: torch.Tensor,
        time_sorted: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the partial log likelihood for the Cox proportional hazards model
        using Breslow's method to handle ties in event time.
        """
        N = len(time_sorted)

        R = [torch.where(time_sorted >= time_sorted[i])[0] for i in range(N)]
        log_denominator = torch.tensor(
            [torch.logsumexp(log_hz_sorted[R[i]], dim=0) for i in range(N)]
        ).to(log_hz_sorted.device)

        return (log_hz_sorted - log_denominator)[event_sorted]
