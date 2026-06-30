import argparse
import copy
import datetime
import json
import os
import random
import re
import time
import numpy as np
from ruamel.yaml import YAML
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path

import utils
from dataset import create_dataset, create_sampler, create_loader, create_eval_dataset, create_eval_loader
from models.model_person_search import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler

ENV_REF_RE = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)")


def read_env_file(env_file, config_path=None):
    if not env_file:
        return {}
    env_path = Path(env_file).expanduser()
    if not env_path.is_absolute():
        candidates = [Path.cwd() / env_path]
        if config_path is not None:
            config_parent = Path(config_path).expanduser().resolve().parent
            candidates.extend([config_parent / env_path, config_parent.parent / env_path])
        env_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not env_path.is_file():
        raise FileNotFoundError(f"env_file does not exist: {env_path}")

    values = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_env_refs(config, env_values):
    def resolve_value(value):
        if isinstance(value, str):
            def replace(match):
                key = match.group(1)
                resolved = os.environ.get(key, env_values.get(key, "")).strip()
                if not resolved:
                    raise ValueError(f"Missing required environment variable: {key}")
                return resolved

            return ENV_REF_RE.sub(replace, value)
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = resolve_value(item)
            return value
        if isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = resolve_value(value[key])
            return value
        return value

    return resolve_value(config)


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pitm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_prd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mrtd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, (image1, image2, text1, text2, idx, replace) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        replace = replace.to(device, non_blocking=True)
        text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
        text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1.0, i / len(data_loader))
        loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd = model(image1, image2, text_input1, text_input2,
                                                                  alpha=alpha, idx=idx, replace=replace)
        loss = 0.
        for j, los in enumerate((loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd)):
            loss += config['weights'][j] * los
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_pitm=loss_pitm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_prd=loss_prd.item())
        metric_logger.update(loss_mrtd=loss_mrtd.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    eval_dataset = data_loader.dataset
    texts = eval_dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)
        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.cpu())
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())
        del text_output, text_input, text_feat, text_embed
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = F.normalize(model.vision_proj(image_feat[:, 0, :]), dim=-1)
        image_embeds.append(image_embed.cpu())
        del image, image_feat, image_embed
    image_embeds = torch.cat(image_embeds, dim=0)
    sims_matrix = text_embeds @ image_embeds.t()
    del text_embeds
    score_matrix_t2i = torch.full((len(texts), len(eval_dataset.image)), -100.0)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    k_test = config['k_test']
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        topk_images = torch.stack([eval_dataset[int(idx)][0] for idx in topk_idx.tolist()]).to(device)
        encoder_output = model.visual_encoder(topk_images)
        text_idx = start + i
        text_feat = text_feats[text_idx:text_idx + 1].to(device)
        text_att = text_atts[text_idx:text_idx + 1].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        output = model.text_encoder.bert(
            encoder_embeds=text_feat.repeat(k_test, 1, 1),
            attention_mask=text_att.repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion',
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1].cpu()
        score_matrix_t2i[text_idx, topk_idx] = score
        del topk_images, encoder_output, output, score, text_feat, text_att
    if args.distributed:
        dist.barrier()
        score_matrix_t2i = score_matrix_t2i.to(device)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item()
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       }
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    print(args)
    print(config)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    # Dataset
    print("Creating retrieval dataset")
    if args.evaluate:
        test_dataset = create_eval_dataset(config)
        test_loader = create_eval_loader(
            test_dataset,
            batch_size=config['batch_size_test'],
            num_workers=0,
        )
        train_loader = val_loader = None
    else:
        train_dataset, val_dataset, test_dataset = create_dataset('ps', config)
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                              batch_size=[config['batch_size_train']] + [
                                                                  config['batch_size_test']] * 2,
                                                              num_workers=[4, 4, 4],
                                                              is_trains=[True, False, False],
                                                              collate_fns=[None, None, None])
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_log = ''

    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)
    optimizer = None
    lr_scheduler = None
    if not args.evaluate:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        if args.resume:
            if optimizer is None or lr_scheduler is None:
                raise RuntimeError("resume requires optimizer state")
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']
        else:
            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)
        if epoch >= config['eval_epoch'] or args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_t2i, test_dataset.img2person, test_dataset.txt2person, args.eval_mAP)
                print('Test:', test_result, '\n')
                if args.evaluate:
                    log_stats = {'epoch': epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()}
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    log_stats = {'epoch': epoch,
                                 **{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'test_{k}': v for k, v in test_result.items()},
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_epoch%02d.pth' % epoch))
                    if test_result['r1'] > best:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = test_result['r1']
                        best_epoch = epoch
                        best_log = log_stats
        if args.evaluate:
            break
        dist.barrier()
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch: {best_epoch} / {max_epoch}\n")
            f.write(f"{best_log}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', default='output/cuhk-pedes')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_mAP', action='store_true', help='whether to evaluate mAP')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    yaml_loader = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml_loader.load(f)
    raw_config = copy.deepcopy(config)
    env_values = read_env_file(config.get('env_file', ''), args.config)
    config = resolve_env_refs(config, env_values)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml_loader.dump(raw_config, f)
    main(args, config)
