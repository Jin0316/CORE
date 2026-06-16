"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from collections import defaultdict  # NEW


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg, ckpt_path, cbl_ckpt_path, n_experts = 20):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config, ckpt_path, cbl_ckpt_path, n_experts)

    def build_datasets(self, cfg, task_id, task_info = None):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        self.task_id = task_id
        self.task_info = None
        if task_info != None: 
            self.task_info = task_info
            print(f'[DATA] Task info specified, current task info: {self.task_info}')
        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(task_id)
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        # print(f'base_task : self.task_info : {self.task_info}')
        if self.task_info != None: 
            # loss = model(samples, task_info = self.task_info)["loss"]
            loss = model(samples, task_info = self.task_info)
            ce_loss = loss["loss"]
            rec_loss = loss["rec_loss"]
            reg_loss = loss["reg_loss"]
        else: 
            loss = model(samples)
            ce_loss = loss["loss"]
            rec_loss = loss["rec_loss"]
            reg_loss = loss["reg_loss"]
        return ce_loss, rec_loss, reg_loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
            
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("ce_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("rec_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("reg_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("total_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)


        # my_iter = len(data_loader.loaders[0])
        my_iter = iters_per_epoch
        for i in metric_logger.log_every(range(my_iter), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            # if i >= iters_per_epoch:
            #     break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                ce_loss, rec_loss, reg_loss = self.train_step(model=model, samples=samples)
                rec_loss = rec_loss.to(ce_loss.device)
                reg_loss = reg_loss.to(ce_loss.device)
                total_loss = ce_loss + rec_loss + reg_loss
            # after_train_step()
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            frozen_indices = model.router.get_frozen_expert_indices()
            # print(frozen_indices)
            if len(frozen_indices) != 0: 
                for expert_idx in frozen_indices:
                    for param in model.experts[expert_idx].parameters():
                        if param.grad is not None:
                            param.grad.zero_()
            
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)
            # for n, p in model.named_parameters():
            #     if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            #         print(f"[non-finite grad] {n} | max={p.grad.abs().max().item():.2e}", flush=True)
                
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(ce_loss=ce_loss.item())
            metric_logger.update(rec_loss=rec_loss.item())
            metric_logger.update(reg_loss=reg_loss.item())
            metric_logger.update(total_loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
        
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("[DATA] result file saved to %s" % final_result_file)

        return final_result_file


    def extract_embeddings_epoch(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        start_iters=None,
        cuda_enabled=False,
    ):
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        logging.info("Extracting Multimodal Embedding.")

        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch

        my_iter = len(data_loader.loaders[0])
        # if my_iter > 400:
        #     my_iter = 200
        print(f'[DATA] Extracting Embedding with iteration: {my_iter}/{len(data_loader.loaders[0])}')

        if (hasattr(model, 'use_external_cbl') and model.use_external_cbl and 
            hasattr(model, 'external_cbl') and model.external_cbl is None):
            
            print("[DATA] Setting up external CBL...")
            # 외부 CBL 함수 import 및 생성
            from minigpt4.models.mini_gpt4 import create_external_cbl
            model.external_cbl = create_external_cbl(
                getattr(model, 'vision_device')
            )

        # Store each sample's data
        sample_data = []

        with torch.no_grad():
            for i in range(my_iter):
                samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )

                keywords  = samples.get("keyword", None)
                image_ids = samples.get("image_id", None)
                questions = samples.get("question", None)
                category  = samples.get("category", None)
                image_embeds, question_embeds = self.extract_embeddings(model=model, samples=samples)

                img_cpu = image_embeds.detach().cpu()
                txt_cpu = question_embeds.detach().cpu()

                # Match each sample's data
                batch_size = img_cpu.size(0)
                for idx in range(batch_size):
                    sample_dict = {
                        'image_embed': img_cpu[idx],
                        'text_embed': txt_cpu[idx], 
                        'keyword': keywords[idx], 
                        'category': category[idx], 
                    }
                    sample_data.append(sample_dict)
                    
        return sample_data

    def extract_embeddings(self, model, samples): 
        image_embeds, question_embeds = model.get_mm_embeds(samples)
        return image_embeds, question_embeds
