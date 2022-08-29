import os
from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer

import logging
log = logging.getLogger(__name__)

VER = 0

class SaneBPE:
    def __init__(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel()
        self.tokenizer = tokenizer

    def train_or_load(self, name, text_iter=None, n_tokens=10000, cache_path='/tmp'):
        dname = os.path.join(cache_path, '_mlucy_utils')
        if not os.path.exists(dname):
            os.mkdir(dname)
        fname = os.path.join(dname, f'{name}.bpe-{VER}.json')
        if os.path.exists(fname):
            self.tokenizer.load(fname)
            log.debug(f'SaneBPE: loaded from {fname}.')
            return

        trainer = BpeTrainer(
            vocab_size=n_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        if text_iter is None:
            log.debug(f'SaneBPE: training on wikipedia.')
            wikipedia = load_dataset('wikipedia', '20220301.en')
            def wiki_text_iter():
                for i in range(1000):
                    yield wikipedia
            text_iter = wiki_text_iter
        else:
            log.debug(f'SaneBPE: training on user text.')
        self.tokenizer.train_from_iterator(text_iter)

        log.debug(f'SaneBPE: saving to {fname}.')
        self.tokenizer.save(fname)


    def _encode(self, x, output):
        assert output in ['ids', 'tokens']
        res = self.tokenizer.encode_batch(x)
        return [getattr(x, output) for x in res]

    def encode(self, x):
        return self._encode(x, 'ids')

    def tokenize(self, x):
        return self._encode(x, 'tokens')

    def decode(self, x):
        return self.tokenizer.decode_batch(x)

def default_lr_scheduler_constructor(optim, epoch):
    return torch.optim.lr_scheduler.MultiplicativeLR(
        optim, lr_lambda=1, last_epoch=epoch)
class Trainer:
    def __init__(self, per_epoch_lr_constructor=None, epoch_lr=None):

        self.scaler = torch.cuda.amp.GradScaler()

        self.epoch = 0
        self.step = 0
        self.epoch_step = 0


        if per_epoch_lr_constructor is None:
            per_epoch_lr_constructor = default_lr_scheduler_constructor
        self.per_epoch_lr = per_epoch_lr_constructor(self.optim, self.epoch)
        if epoch_lr is None:
            epoch_lr = default_lr_scheduler_constructor(self.optim, self.epoch)
        self.epoch_lr = epoch_lr

    def backward(self, loss):
        c = self.config
        if c.autocast:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        if c.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), c.clip_grad_norm)

        if c.autocast:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

    def train_one_step(self, batch_x, batch_y):
        c = self.config
        with torch.cuda.amp.autocast(enabled=c.autocast):
            model_y = self.model(batch_x)
            loss = self.loss(model_y, batch_y)
            self.step += 1
            self.epoch_step += 1

            to_log_default = {
                'loss': loss,
                'lr': self.per_epoch_lr.get_last_lr(),
            }
            with torch.no_grad():
                res = self.to_log(model_y, batch_y, to_log_default)

            self.backward(loss)
            self.per_epoch_lr.step()
        return res

    def val_one_step(self, batch_x, batch_y):
        c = self.config
        with torch.cuda.amp.autocast(enabled=c.autocast):
            with torch.no_grad():
                model_y = self.model(batch_x)
                loss = self.loss(model_y, batch_y)
                to_log_default = {
                    'loss': loss,
                    'lr': self.per_epoch_lr.get_last_lr(),
                }
                return self.to_log(model_y, batch_y, to_log_default, val=True)


    def train_one_epoch(self, validate=True):
        c = self.config
        train_data_iter, val_data_iter, _ = self.new_data_iters(c)

        self.model.train(True)
        for _batch_x, _batch_y in train_data_iter:
            self.epoch_step = 0
            self.per_epoch_lr = self.per_epoch_lr_constructor(self.epoch)

            batch_x, batch_y = _batch_x.to(c.device), _batch_y.to(c.device)
            to_log = self.train_one_step(batch_x, batch_y)
            self.log(to_log)

        if validate:
            self.model.train(False)
            to_merge = {}
            for _batch_x, _batch_y in val_data_iter:
                batch_x, batch_y = _batch_x.to(c.device), _batch_y.to(c.device)
                to_log = self.val_one_step(batch_x, batch_y)
                for k, v in to_log.items():
                    acc = to_merge.setdefault(k, [])
                    acc.append(v)
                self.log(dict((k, np.mean(v)) for k, v in to_merge.items()))

        self.epoch += 1
        self.epoch_lr.step()