import datetime
import itertools
from collections import defaultdict
from pathlib import Path
import time

import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline,
                       PNDMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import Repository
from PIL.Image import Resampling
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import BertTokenizer, CLIPFeatureExtractor, CLIPTokenizer

from clip_scores import CLIPEvaluator, select_init
from lpips import LPIPS

from optimizers import get_optimizer
from templates import *
from text_emb import (TextualInversionCLIPTextModel,
                      TextualInversionLDMBertModel)
from utils import *


def main():
    args = parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logging_dir = f"{args.output_dir}/{args.logging_dir}-{now}"
    if not args.pretrained_model_name_or_path:
        args.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" if args.model == "sd" \
            else "CompVis/ldm-text2im-large-256"
    if not args.resolution:
        args.resolution = 256 if args.model == "ldm" else 512
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    use_auth_token = not args.offline_mode
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if args.logger == "tensorboard":
            stat_logger = SummaryWriter(logging_dir)
        elif args.logger == "wandb":
            stat_logger = "wandb"
            if args.project_name:
                name = args.project_name
            elif args.variant == "vanilla":
                name = "vanilla-baseline"
            elif args.variant == "clip_early_stopping":
                name = "clip-baseline"
            else:
                name = "ours"
            wandb.init(entity=args.wandb, project=name)
            if args.exp_name:
                wandb.run.name = args.exp_name
            else:
                concept = args.train_data_dir.split("/")[-1]
                if concept == "train":
                    concept = args.train_data_dir.split("/")[-2]
                wandb.run.name = f"{concept}-{args.model}-{args.init_strategy}-{args.optimizer}-{args.template_set}"
    accelerator.init_trackers("")

    # Load the tokenizer and add the placeholder token as an additional special token
    if args.model == "sd":
        if args.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        elif args.pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                      subfolder="tokenizer", use_auth_token=use_auth_token)
    else:
        if args.tokenizer_name:
            tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        elif args.pretrained_model_name_or_path:
            tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                      subfolder="tokenizer", use_auth_token=use_auth_token)
    if args.init_strategy == "manual":
        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
    else:
        initializer_token_id = select_init(args.train_data_dir, tokenizer, strategy=args.init_strategy)

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    if args.model == "sd":
        text_encoder = TextualInversionCLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                     subfolder="text_encoder", use_auth_token=use_auth_token)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder="vae", use_auth_token=use_auth_token)
    else:
        text_encoder = TextualInversionLDMBertModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                    subfolder="bert")
        max_position_ids = text_encoder.model.embed_positions.weight.size()[0]
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder="vqvae", use_auth_token=use_auth_token)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                subfolder="unet", use_auth_token=use_auth_token)

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    text_encoder.patch_emb(token_embeds[initializer_token_id])
    original_embed = token_embeds[initializer_token_id].to(accelerator.device)

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    if args.model == "sd":
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
    else:
        params_to_freeze = itertools.chain(
            text_encoder.model.embed_positions.parameters(),
            text_encoder.model.layers.parameters(),
            text_encoder.to_logits.parameters(),
        )
    freeze_params(params_to_freeze)

    args.batch_size = args.train_batch_size
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    args.optimizer = args.optimizer.lower()
    optimizer_params = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'adam_epsilon': args.adam_epsilon,
        'sam_momentum': args.sam_momentum,
        'sam_rho': args.sam_rho,
        'sam_adaptive': args.sam_adaptive,
    }
    optimizer = get_optimizer(model=text_encoder.get_input_embeddings(),
                              name=args.optimizer, opt_args=optimizer_params)

    if args.model == "sd":
        noise_scheduler = PNDMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    use_auth_token=use_auth_token)
    else:
        noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    alphas = noise_scheduler.alphas
    betas = noise_scheduler.betas
    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas_cumprod_prev = torch.cat((torch.tensor([1.]), alphas_cumprod[:-1]))

    posterior_variance = (1 - args.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + args.v_posterior * betas
    lvlb_weights = betas ** 2 / (2 * posterior_variance * alphas * (1 - alphas_cumprod))
    lvlb_weights[0] = lvlb_weights[1]

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
        template_set=args.template_set,
        flip_p=args.flip_p,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.optimizer not in ["sam"]:
        text_encoder, vae, unet, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, vae, unet, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        text_encoder, vae, unet, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, vae, unet, train_dataloader, eval_dataloader, lr_scheduler
        )

    # Move vae and unet to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    lvlb_weights = lvlb_weights.to(accelerator.device)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # log train data
    if accelerator.is_main_process:
        os.makedirs(logging_dir)
        if args.logger == "tensorboard":
            train_images = [Image.open(x).resize((args.resolution, args.resolution), resample=Resampling.BICUBIC)
                            for x in train_dataset.image_paths]
            grid = log_images(train_images, name="train", logging_dir=logging_dir, step=0)
            stat_logger.add_image("train", np.array(grid).transpose(2, 0, 1), 0)
        elif args.logger == "wandb":
            dataset_log(args.train_data_dir)

    args.vanilla = args.variant == "vanilla"
    args.early_stopping = args.variant in ["ours_early_stopping", "clip_early_stopping"]

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    val_prompts = [x.format(args.placeholder_token) for x in
                   np.random.choice(imagenet_templates_base, args.n_val_prompts)]
    fixed_input = next(iter(train_dataloader))["pixel_values"]

    # initialize clip for samples evaluation
    clip = CLIPEvaluator(device=accelerator.device)

    # initialize the LPIPS scorer
    lpips = LPIPS(net="alex").to(accelerator.device)

    with open(f"{logging_dir}/val_prompts.txt", "w") as f:
        f.writelines("\n".join(val_prompts))

    # Create the pipeline using the trained modules
    if accelerator.is_main_process:
        if args.model == "sd":
            if args.pretrained_model_name_or_path:
                safety_checker_path = args.pretrained_model_name_or_path
            else:
                safety_checker_path = "CompVis/stable-diffusion-safety-checker"
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker_path,
                                                                          subfolder="safety_checker",
                                                                          use_auth_token=use_auth_token)
            feature_extractor_path = args.pretrained_model_name_or_path if args.offline_mode else \
                "openai/clip-vit-base-patch32"
            feature_extractor = CLIPFeatureExtractor.from_pretrained(feature_extractor_path,
                                                                     subfolder="feature_extractor",
                                                                     use_auth_token=use_auth_token)
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            pipeline.safety_checker.to(accelerator.device)
        else:
            pipeline = LDMTextToImagePipeline(
                vqvae=vae,
                bert=accelerator.unwrap_model(text_encoder),
                unet=unet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
            )
        if args.save_init_embeds:
            name = "initial_embeds.bin"
            save_progress(original_embed.unsqueeze(0), args.placeholder_token, logging_dir, name=name)
        if args.save_pipeline:
            if not args.pipeline_output_dir:
                args.pipeline_output_dir = args.output_dir
            pipeline.save_pretrained(args.pipeline_output_dir)
        if not args.no_sample_before_start:
            # sample score and log images
            set_seed(args.sampling_seed)
            sample(prompts=val_prompts, pipe=pipeline, clip=clip, lpips=lpips, logger=stat_logger, input=fixed_input,
                   logging_dir=logging_dir, bs=args.batch_size, sample_steps=args.sample_steps,
                   guidance=args.guidance, step=global_step, log_unscaled=args.log_unscaled, is_vanilla=args.vanilla,
                   fp16=args.mixed_precision == "fp16"
                   )
            if args.seed is not None:
                set_seed(args.seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    wandb.config.update(args)
    logs = {}
    if args.deterministic:
        # all_timesteps = torch.arange(1000, device=accelerator.device)[torch.randperm(1000)]
        all_timesteps = torch.tensor([700, 850, 950], device=accelerator.device).long()

    if args.variant == "clip_early_stopping":
        early_stopper = ClipEarlyStopper(eps=args.early_stop_eps, patience=args.early_stop_patience)
    elif args.variant == "ours_early_stopping":
        print("STOPPING PREPARE")
        early_stopper = VarEarlyStopper(eps=args.early_stop_eps, window=args.early_stop_patience)

        with torch.inference_mode(), torch.random.fork_rng(devices=[accelerator.device, 'cpu']):
            eval_batch = next(iter(eval_dataloader))
            if args.model == "ldm":
                    eval_batch["input_ids"] = eval_batch["input_ids"][:, :max_position_ids]
            eval_latents = vae.encode(eval_batch["pixel_values"]).latent_dist.sample().detach()
            eval_latents = eval_latents * 0.18215
            eval_noise = torch.randn(eval_latents.shape).to(eval_latents.device)
            eval_timesteps = torch.randperm(1000)[:eval_latents.shape[0]].to(eval_latents.device)
            eval_noisy_latents = noise_scheduler.add_noise(eval_latents, eval_noise, eval_timesteps)
    else:
        early_stopper = None

    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        epoch_stats = defaultdict(float)
        for step, batch in enumerate(train_dataloader):
            if (args.early_stopping and early_stopper.stopped) or global_step >= args.max_train_steps:
                break
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                if args.deterministic:
                    # doesn't work properly with accumulation
                    # timesteps = all_timesteps[step * bsz % 1000: (step + 1) * bsz % 1000]
                    # timesteps = all_timesteps[:bsz]
                    # timesteps = all_timesteps[0].repeat(bsz)
                    # timesteps = torch.tensor([100, 900], device=accelerator.device).repeat(bsz // 2)
                    timesteps = all_timesteps[torch.arange(step * bsz, (step + 1) * bsz) % len(all_timesteps)]
                    if step < 2:
                        print(timesteps)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.model == "ldm":
                    batch["input_ids"] = batch["input_ids"][:, :max_position_ids]
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                def closure():
                    optimizer.zero_grad()
                    mse_loss = F.mse_loss(noise, noise_pred, reduction="none").mean([1, 2, 3])
                    loss = (1 - args.original_elbo_weight) * mse_loss.mean()

                    loss_vlb = (lvlb_weights[timesteps] * mse_loss).mean()
                    loss += args.original_elbo_weight * loss_vlb

                    logs['loss'] = loss.detach()
                    logs['loss_rec'] = mse_loss.detach().mean()
                    logs['loss_vlb'] = loss_vlb.detach()
                    accelerator.backward(loss, retain_graph=True)

                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().concept_token.grad
                    else:
                        grads = text_encoder.get_input_embeddings().concept_token.grad

                    logs['gradient_norm'] = torch.norm(grads, 2).detach()

                    return loss

                closure()

                if args.optimizer == "sam":
                    optimizer.step(closure)
                else:
                    optimizer.step()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                if args.variant == "ours_early_stopping" and global_step % args.early_stop_freq == 0:
                    with torch.inference_mode():
                        eval_encoder_hidden_states = text_encoder(eval_batch["input_ids"])[0]
                        eval_noise_pred = unet(eval_noisy_latents, eval_timesteps, eval_encoder_hidden_states).sample
                        eval_mse_loss = F.mse_loss(eval_noise, eval_noise_pred, reduction="none").mean().item()
                        logs["normalized_var"] = early_stopper(eval_mse_loss)
                        # logs[f"eval_loss_{'_'.join(map(str, eval_timesteps.cpu().numpy()))}"] = eval_mse_loss
                        logs["eval_loss"] = eval_mse_loss

                if global_step % args.sample_frequency == 0 and args.variant not in ["ours_early_stopping", "short_iters"]:
                    set_seed(args.sampling_seed)
                    cur_clip = sample(prompts=val_prompts, pipe=pipeline, clip=clip, lpips=lpips,
                                      logger=stat_logger, input=fixed_input, bs=args.batch_size,
                                      sample_steps=args.sample_steps, guidance=args.guidance,
                                      logging_dir=logging_dir, step=max(1, global_step),
                                      log_unscaled=args.log_unscaled, is_vanilla=args.vanilla,
                                      fp16=args.mixed_precision == "fp16",
                                      )
                    if args.variant == "clip_early_stopping":
                        early_stopper(cur_clip)

                global_step += 1
                set_seed(args.seed + global_step)

            learned_embed = text_encoder.get_input_embeddings().concept_token.data

            if global_step % args.save_steps == 0:
                name = f"learned_embeds_{global_step:04d}.bin"
                save_progress(learned_embed, args.placeholder_token, logging_dir, name=name)

            logs["lr"] = lr_scheduler.get_last_lr()[0]

            logs["emb_reg_loss"] = torch.norm(original_embed - learned_embed, 2).detach()

            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            for k in logs:
                if k != "lr":
                    epoch_stats[k] += logs[k]
                if args.logger == "tensorboard":
                    stat_logger.add_scalar(k, logs[k], global_step)
            if args.logger == "wandb":
                wandb.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        epoch_stats = {f"{k}_epoch": v / (step + 1) for k, v in epoch_stats.items()}
        if args.logger == "wandb":
            wandb.log(epoch_stats, step=epoch)
        else:
            for k in epoch_stats:
                stat_logger.add_scalar(f"{k}_epoch", epoch_stats[k], epoch)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # save last trained embeddings
            name = f"learned_embeds_last.bin"
            save_progress(learned_embed, args.placeholder_token, logging_dir, name)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

    if args.variant in ['vanilla', 'ours_early_stopping', "short_iters"]:
        best_clip, best_iter = 0, 0
        trash_dir = os.path.join(logging_dir, "afterscore")
        os.makedirs(trash_dir)
        last_step = args.max_train_steps if args.variant in ['vanilla', "short_iters"] else global_step + 1
        # iterate over saved embeds
        start = time.time()
        for embed in sorted(os.listdir(os.path.join(logging_dir, "embeds"))):
            if embed == "initial_embeds.bin":
                cur_iter = last_step
            elif embed == "learned_embeds_last.bin":
                cur_iter = 2 * last_step
            else:
                cur_iter = int(embed.split("learned_embeds_")[1][:4]) + last_step

            embed_path = os.path.join(logging_dir, "embeds", embed)
            if (args.variant == "short_iters" and embed == "learned_embeds_last.bin") or args.variant != "short_iters":
                embed = torch.load(embed_path)[args.placeholder_token]

                text_encoder.get_input_embeddings().concept_token.data = embed
                # sample, calc clip_img_score
                set_seed(args.sampling_seed)
                cur_clip = sample(prompts=val_prompts, pipe=pipeline, clip=clip, lpips=lpips, logger=stat_logger,
                                  input=fixed_input, logging_dir=trash_dir, bs=args.batch_size,
                                  sample_steps=args.sample_steps, guidance=args.guidance, step=cur_iter,
                                  log_unscaled=False, is_vanilla=False, afterscore=True,
                                  fp16=args.mixed_precision == "fp16",
                                  )
                if cur_clip > best_clip:
                    best_clip = cur_clip
                    best_iter = cur_iter
        # report best
        if stat_logger == "wandb":
            wandb.run.summary["AFTERSCORE TIME, s"] = (time.time() - start) * 1000
            wandb.run.summary["BEST CLIP"] = best_clip
            wandb.run.summary["BEST EMBEDDING"] = best_iter - last_step
        else:
            stat_logger.add_scalar("best_clip", best_clip)
            stat_logger.add_scalar("best_embed", best_iter - last_step)


if __name__ == "__main__":
    main()
