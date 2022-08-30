import torch
import re

class ParamSpec:
    def __init__(self, mn, m, pn, p):
        self.mn = mn
        self.m = m
        self.pn = pn
        self.p = p

    def module_name(self):
        return self.mn

    def module(self):
        return self.m

    def parameter_name(self):
        return self.pn

    def parameter(self):
        return self.p

    def is_bias(self):
        return re.fullmatch('.*bias', self.pn)

    def is_weight(self):
        return re.fullmatch('.*weight', self.pn)

    def is_normalization(self):
        return self.m.__class__.__module__ == torch.nn.modules.normalization.__name__

    def is_embedding(self):
        return self.m.__class__.__module__ == torch.nn.modules.sparse.__name__

def adamw_easy_grouper(pspec):
    if pspec.is_bias() or pspec.is_normalization() or pspec.is_embedding():
        return 'no_decay'
    return 'decay'

def _group_params(obj, grouper, out, output, mn):
    for new_mn, m in obj.named_children():
        _group_params(m, grouper, out, output, mn=('' if mn == '' else mn+'.')+new_mn)
    for pn, p in obj._parameters.items():
        if p is None:
            continue
        group = grouper(ParamSpec(mn, obj, pn, p))
        if output == 'params':
            out.setdefault(group, []).append(p)
        elif output == 'names':
            out.setdefault(group, []).append(mn+'.'+pn)

def group_params(obj, grouper, output='params'):
    assert output in ['params', 'names']
    out = {}
    _group_params(obj, grouper, out, output, mn='')
    RSI: fix
    assert len(list(obj.parameters())) == sum([len(x) for x in out.values()])
    return out

def iter_params(obj, f):
    return group_params(obj, f, output='params')
