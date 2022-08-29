import os
from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer
from .config import Config

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

class TrainerConfig(Config):
    def __init__(self, obj, presets):
        super().__init__(
            obj,
            presets,
            required=[
            ],
            preset_map={
            },
            defaults={
                'autocast': True,
                'clip_grad_norm': None,

                'save_every': None,
                'save_best': 'val_loss',

                'optim': putils.AdamW,
                'optim_grouper': putils.adamw_easy_grouper,
                'optim_args': {'weight_decay': {'decay': 1e-3, 'no_decay': 0}},

                'epoch_lr_scheduler': torch.optim.lr_scheduler.MultiplicativeLR
                'epoch_lr_scheduler_args': {'lr_lambda': 1},
                'per_epoch_lr_scheduler': torch.optim.lr_scheduler.MultiplicativeLR
                'per_epoch_lr_scheduler_args': {'lr_lambda': 1},
            },
        )

class Trainer:
    def __init__(self, model, config):
        self.model = model
        c = self.config = config

        self.scaler = torch.cuda.amp.GradScaler()

        self.epoch = 0
        self.step = 0
        self.examples = 0
        self.epoch_step = 0

        self.last_saved = self.step
        self.best_score = None

        self.optim = self.gen_optim()
        self.epoch_lr = self.epoch_lr_scheduler()
        self.per_epoch_lr = self.per_epoch_lr_scheduler()

    # RSI: pick up here:
    # Separate load into resume and fine-tune
    # Actually add tensorboard logic

    def log_extra(self, batch_x, model_y, batch_y):
        return {}

    def tb_log(self, obj):
        if val is False:
            res = {'loss': loss, 'lr': self.per_epoch_lr.get_last_lr()}
        else:
            res = {'val_loss': loss}

        return res

    def save_all(self, path=None):
        to_save = {}
        for k, v in self.__dict__.items():
            if callable(getattr(v, 'state_dict', None)):
                to_save[k] = v.state_dict()
            else:
                to_save[k] = v

        if path is None:
            path = self.resume_path()
        torch.save(to_save, path)

    def load_all(self, path=None):
        if path is None:
            path = self.resume_path()
        obj = torch.load(path)

        for k, v in obj.items():
            if callable(getattr(getattr(self, k), 'state_dict', None)):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)

    def gen_optim(self):
        c = self.config
        if c.optim_grouper is None:
            params = {'': self.model.parameters()}
        else:
            params = putils.group_params(self.model, c.optim_grouper)
        split_opts = dict((k, {}) for k, v in params.items())
        all_opts = {}
        for arg, vals in c.optim_args:
            if isinstance(vals, dict):
                for k, v in vals:
                    if k in obj:
                        split_opts[k][arg] = v
            else:
                all_opts[arg] = vals
        split_params = [{'params': params[k], **split_opts[k]} for k in params]
        return c.optim(split_params, **all_opts)

    def per_epoch_lr_scheduler(self):
        c = self.config.optim
        return c.per_epoch_lr_scheduler(
            self.optim, last_epoch=self.epoch-1, **c.per_epoch_lr_scheduler_args)

    def epoch_lr_scheduler(self):
        c = self.config
        return c.epoch_lr_scheduler(
            self.optim, last_epoch=self.epoch-1, **c.epoch_lr_scheduler_args)

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
            self.examples += batch_x.shape[0]

            res = {
                'loss': loss,
                'lr': self.per_epoch_lr.get_last_lr(),
                **self.log_extra(batch_x, batch_y),
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
                res = {
                    'val_loss': loss,
                    **self.log_extra(batch_x, model_y, batch_y),
                }
                return res

    def train_one_epoch(self, validate=True):
        c = self.config
        train_data_iter, val_data_iter, _ = self.new_data_iters(c)

        self.model.train(True)
        for _batch_x, _batch_y in train_data_iter:
            self.epoch_step = 0
            self.per_epoch_lr = self.per_epoch_lr_constructor(self.epoch)

            batch_x, batch_y = _batch_x.to(c.device), _batch_y.to(c.device)
            to_log = self.train_one_step(batch_x, model_y, batch_y)
            self.tb_log(to_log)

            if c.save_every is not None and c.save_every < self.last_saved - self.step:
                self.last_saved = self.step
                self.save_all('step_chk')

        self.epoch += 1
        self.epoch_lr.step()

        self.save_all('epoch_chk')

        if validate:
            self.model.train(False)
            to_merge = {}
            for _batch_x, _batch_y in val_data_iter:
                batch_x, batch_y = _batch_x.to(c.device), _batch_y.to(c.device)
                to_log = self.val_one_step(batch_x, batch_y)
                for k, v in to_log.items():
                    acc = to_merge.setdefault(k, [])
                    acc.append(v)
                self.tb_log(dict((k, np.mean(v)) for k, v in to_merge.items()))

            if c.save_on_best is not None:
                score = to_log[c.save_best]
                if self.best_score is None or score < self.best_score:
                    self.best_score = score
                    self.save_all(f'best_{self.epoch:04d}')

