import torch
import re

def adamw_easy_grouper(module_name, module, parameter_name, parameter):
    if module.__class__.__module__ == torch.nn.modules.normalization.__name__:
        return 'no_decay'
    if re.fullmatch('.*bias', parameter_name):
        return 'no_decay'
    return 'decay'

def _group_params(obj, grouper, out, output, mn):
    for new_mn, m in obj.named_children():
        _group_params(m, grouper, out, output, mn=('' if mn == '' else mn+'.')+new_mn)
    for pn, p in obj._parameters.items():
        group = grouper(module_name=mn, module=obj, parameter_name=pn, parameter=p)
        if output == 'params':
            out.setdefault(group, []).append(p)
        elif output == 'names':
            out.setdefault(group, []).append(mn+'.'+pn)

def group_params(obj, grouper, output='params'):
    assert output in ['params', 'names']
    out = {}
    _group_params(obj, grouper, out, output, mn='')
    assert len(list(obj.parameters())) == sum([len(x) for x in out.values()])
    return out


