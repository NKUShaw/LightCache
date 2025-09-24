from typing import Optional, Tuple, Union, Dict, Any
from .memory import MemoryEfficientMixin, round_func
import numpy as np
import torch

from diffusers.utils import logging, deprecate
from diffusers.models.resnet import SpatioTemporalResBlock, ResnetBlock2D
from diffusers.models.transformers.transformer_temporal import (TransformerSpatioTemporalModel, TransformerTemporalModelOutput, TransformerTemporalModel)
from diffusers.models.attention import TemporalBasicTransformerBlock, _chunked_feed_forward
from diffusers.models.unets import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.models.unets.unet_3d_blocks import (DownBlockSpatioTemporal, CrossAttnDownBlockSpatioTemporal, UpBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal)
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class BatchUNetSpatioTemporalConditionModel(MemoryEfficientMixin, UNetSpatioTemporalConditionModel):
    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            added_time_ids: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        # print(f"BatchUNetSpatioTemporalConditionModel Shape: {sample.shape}")
        chunk_size = max(round_func(sample.size(0) / self.num_chunk), 1)                     # batch_size * num_frames / num_chunk
        chunk_sample_cache = []
        for i in range(0, sample.size(0), chunk_size):
            chunk_sample = self.conv_norm_out(sample[i:i + chunk_size])
            chunk_sample = self.conv_act(chunk_sample)
            chunk_sample = self.conv_out(chunk_sample)
            chunk_sample_cache.append(chunk_sample)
        sample = torch.cat(chunk_sample_cache)
        
        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)

class BatchSpatioTemporalResBlock(MemoryEfficientMixin, SpatioTemporalResBlock):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_temb = temb[i:i + chunk_size] if temb is not None else temb
            chunk_hidden_states = self.spatial_res_block(hidden_states[i:i + chunk_size], chunk_temb)
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)
        # print(f"BatchSpatioTemporalResBlock Shape: {hidden_states.shape}")
        
        chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(-1), chunk_size):
            chunk_hidden_states = self.temporal_res_block(hidden_states[..., i:i + chunk_size], temb)
            chunk_hidden_states = self.time_mixer(
                x_spatial=hidden_states_mix[..., i:i + chunk_size],
                x_temporal=chunk_hidden_states,
                image_only_indicator=image_only_indicator,
            )
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        return hidden_states


class BatchTransformerSpatioTemporalModel(MemoryEfficientMixin, TransformerSpatioTemporalModel):
    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        # 0. Device
        device = next(self.parameters()).device

        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[None, :].broadcast_to(
            height * width, batch_size, 1, time_context.shape[-1]
        )
        time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        # print(f"BatchTransformerSpatioTemporalModel Shape: {hidden_states.shape}")
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.norm(hidden_states[i:i + chunk_size])
            inner_dim = chunk_hidden_states.shape[1]
            chunk_hidden_states = chunk_hidden_states.permute(0, 2, 3, 1).reshape(chunk_hidden_states.size(0),
                                                                                  height * width, inner_dim)
            chunk_hidden_states = self.proj_in(chunk_hidden_states)
            chunk_hidden_states_cache.append(chunk_hidden_states)

        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            # spatio
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = block(
                    hidden_states[i:i + chunk_size],
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                )
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb.to(hidden_states_mix)

            # temporal
            # In-Block TemporalBasicTransformerBlock Slicing
            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            while hidden_states.shape[0] % chunk_size != 0:
                chunk_size += 1
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.time_mixer(
                    x_spatial=hidden_states[i:i + chunk_size],
                    x_temporal=hidden_states_mix[i:i + chunk_size],
                    image_only_indicator=image_only_indicator[..., i // batch_size:(i + chunk_size) // batch_size],
                )
                if len(self.transformer_blocks) == 1:
                    chunk_hidden_states = self.proj_out(chunk_hidden_states)
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        # 3. Output

        if len(self.transformer_blocks) != 1:
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.proj_out(hidden_states[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class BatchTemporalBasicTransformerBlock(MemoryEfficientMixin, TemporalBasicTransformerBlock):

    def forward(
            self,
            hidden_states: torch.Tensor,
            num_frames: int,
            encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)
        # print(f"BatchTemporalBasicTransformerBlock Shape: {hidden_states.shape}")
        chunk_size = max(round_func(hidden_states.size(0) / self.num_spatial_chunk), 1)
        chunk_hidden_states_cache = []

        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_residual = hidden_states[i:i + chunk_size].cuda()

            chunk_hidden_states = self.norm_in(hidden_states[i:i + chunk_size].cuda())

            if self._chunk_size is not None:
                chunk_hidden_states = _chunked_feed_forward(self.ff_in, chunk_hidden_states, self._chunk_dim,
                                                            self._chunk_size)
            else:
                chunk_hidden_states = self.ff_in(chunk_hidden_states)

            if self.is_res:
                chunk_hidden_states = chunk_hidden_states + chunk_residual

            chunk_norm_hidden_states = self.norm1(chunk_hidden_states)
            chunk_attn_output = self.attn1(chunk_norm_hidden_states, encoder_hidden_states=None)
            chunk_hidden_states = chunk_attn_output + chunk_hidden_states

            # 3. Cross-Attention
            if self.attn2 is not None:
                chunk_norm_hidden_states = self.norm2(chunk_hidden_states)
                chunk_attn_output = self.attn2(chunk_norm_hidden_states,
                                               encoder_hidden_states=encoder_hidden_states[i:i + chunk_size])
                chunk_hidden_states = chunk_attn_output + chunk_hidden_states

            # 4. Feed-forward
            chunk_norm_hidden_states = self.norm3(chunk_hidden_states)

            if self._chunk_size is not None:
                chunk_ff_output = _chunked_feed_forward(self.ff, chunk_norm_hidden_states, self._chunk_dim,
                                                        self._chunk_size)
            else:
                chunk_ff_output = self.ff(chunk_norm_hidden_states)

            if self.is_res:
                chunk_hidden_states = chunk_ff_output + chunk_hidden_states
            else:
                chunk_hidden_states = chunk_ff_output

            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)
        return hidden_states

class BatchDownBlockSpatioTemporal(MemoryEfficientMixin, DownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            output_states = output_states + (hidden_states,)

        # print(f"BatchDownBlockSpatioTemporal Shape: {hidden_states.shape}")
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class BatchCrossAttnDownBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnDownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

            output_states = output_states + (hidden_states,)
        # print(f"BatchCrossAttnDownBlockSpatioTemporal Shape: {hidden_states.shape}")
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)
        return hidden_states, output_states

class BatchUpBlockSpatioTemporal(MemoryEfficientMixin, UpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        # print(f"BatchUpBlockSpatioTemporal Shape: {hidden_states.shape}")
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states

class BatchCrossAttnUpBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnUpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

        # print(f"BatchCrossAttnUpBlockSpatioTemporal Shape: {hidden_states.shape}")
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


from diffusers.models.unets.unet_motion_model import (CrossAttnDownBlockMotion, DownBlockMotion,
                                                      CrossAttnUpBlockMotion, UpBlockMotion,
                                                      UNetMidBlockCrossAttnMotion)

class BatchDownBlockMotion(MemoryEfficientMixin, DownBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            num_frames: int = 1,
            *args,
            **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        blocks = zip(self.resnets, self.motion_modules)
        for resnet, motion_module in blocks:

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class BatchCrossAttnDownBlockMotion(MemoryEfficientMixin, CrossAttnDownBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            additional_residuals: Optional[torch.Tensor] = None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()
        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class BatchUpBlockMotion(MemoryEfficientMixin, UpBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            upsample_size=None,
            num_frames: int = 1,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        blocks = zip(self.resnets, self.motion_modules)

        for resnet, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class BatchCrossAttnUpBlockMotion(MemoryEfficientMixin, CrossAttnUpBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        blocks = zip(self.resnets, self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class BatchUNetMidBlockCrossAttnMotion(MemoryEfficientMixin, UNetMidBlockCrossAttnMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        blocks = zip(self.resnets[:-2], self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.resnets[-1](hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


""" Stable-Video Diffusion Blocks """


class BatchDownBlockSpatioTemporal(MemoryEfficientMixin, DownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class BatchCrossAttnDownBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnDownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class BatchUpBlockSpatioTemporal(MemoryEfficientMixin, UpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class BatchCrossAttnUpBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnUpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states

from diffusers.models import UNet3DConditionModel
from diffusers.models.unets.unet_3d_blocks import CrossAttnDownBlock3D, CrossAttnUpBlock3D, DownBlock3D, UpBlock3D
from diffusers.utils import BaseOutput

class UNet3DConditionOutput(BaseOutput):
    """
    The output of [`UNet3DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor

class BatchUNet3DConditionModel(MemoryEfficientMixin, UNet3DConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[torch.Tensor]]:

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
        )

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)

        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 6. post-process
        # print(sample.shape) torch.Size([50, 320, 32, 32])
        chunk_size = max(round_func(sample.size(0) / self.num_chunk), 1)
        chunk_sample_cache = []
        with torch.no_grad():
            for i in range(0, sample.size(0), chunk_size):
                chunk_sample = self.conv_norm_out(sample[i:i + chunk_size])
                chunk_sample = self.conv_act(chunk_sample)
                chunk_sample = self.conv_out(chunk_sample)
                chunk_sample_cache.append(chunk_sample.detach())
        sample = torch.cat(chunk_sample_cache)
        # print(sample.shape) torch.Size([50, 4, 32, 32])
        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)
        # print(sample.shape)  torch.Size([2, 4, 25, 32, 32])
        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

class BatchCrossAttnDownBlock3D(MemoryEfficientMixin, CrossAttnDownBlock3D):
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Dict[str, Any] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states += (hidden_states,)
        return hidden_states, output_states

class BatchDownBlock3D(MemoryEfficientMixin, DownBlock3D):
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        num_frames: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
            output_states += (hidden_states,)

        return hidden_states, output_states

class BatchUpBlock3D(MemoryEfficientMixin, UpBlock3D):
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        num_frames: int = 1,
    ) -> torch.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states

class BatchCrossAttnUpBlock3D(MemoryEfficientMixin, CrossAttnUpBlock3D):
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Dict[str, Any] = None,
    ) -> torch.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        # TODO(Patrick, William) - attention mask is not used
        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        return hidden_states

class BatchTransformerTemporalModel(MemoryEfficientMixin, TransformerTemporalModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: torch.LongTensor = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> TransformerTemporalModelOutput:
        
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        
        
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        # hidden_states = self.norm(hidden_states)

        
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.norm(hidden_states[i:i + chunk_size])
            inner_dim = chunk_hidden_states.shape[1]
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        # for block in self.transformer_blocks:
        #     hidden_states = block(
        #         hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         timestep=timestep,
        #         cross_attention_kwargs=cross_attention_kwargs,
        #         class_labels=class_labels,
        #     )

        
        for block in self.transformer_blocks:
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                if encoder_hidden_states is not None:
                    chunk_hidden_states = block(
                        hidden_states[i:i + chunk_size],
                        encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )
                else:
                    chunk_hidden_states = block(
                        hidden_states[i:i + chunk_size],
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        # 3. Output
        # hidden_states = self.proj_out(hidden_states)

        
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.proj_out(hidden_states[i:i + chunk_size])
            chunk_hidden_states_cache.append(chunk_hidden_states)

        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class BatchResnetBlock2D(MemoryEfficientMixin, ResnetBlock2D):
    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor


        # hidden_states = self.norm1(hidden_states)
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.norm1(hidden_states[i:i + chunk_size])
            chunk_hidden_states_cache.append(chunk_hidden_states)

        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            # hidden_states = self.norm2(hidden_states)
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.norm2(hidden_states[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)
    
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            # hidden_states = self.norm2(hidden_states)
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.norm2(hidden_states[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)
    
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            # hidden_states = self.norm2(hidden_states)
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.norm2(hidden_states[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

from diffusers.models.transformers.transformer_2d import Transformer2DModel

class BatchTransformer2DModel(MemoryEfficientMixin, Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                )
            else:
                # hidden_states = block(
                #     hidden_states,
                #     attention_mask=attention_mask,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     timestep=timestep,
                #     cross_attention_kwargs=cross_attention_kwargs,
                #     class_labels=class_labels,
                # )
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = block(
                        hidden_states[i:i + chunk_size],
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
                
        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

from diffusers.models.transformers.transformer_easyanimate import EasyAnimateTransformer3DModel

class BatchEasyAnimateTransformer3DModel(MemoryEfficientMixin, EasyAnimateTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        batch_size, channels, video_length, height, width = hidden_states.size()
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. Time embedding
        temb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        temb = self.time_embedding(temb, timestep_cond)
        image_rotary_emb = self.rope_embedding(hidden_states)

        # 2. Patch embedding
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 1)
        if control_latents is not None:
            hidden_states = torch.concat([hidden_states, control_latents], 1)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, F, H, W] -> [BF, C, H, W]
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
            0, 2, 1, 3, 4
        )  # [BF, C, H, W] -> [B, F, C, H, W]
        hidden_states = hidden_states.flatten(2, 4).transpose(1, 2)  # [B, F, C, H, W] -> [B, FHW, C]

        # 3. Text embedding
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        if encoder_hidden_states_t5 is not None:
            encoder_hidden_states_t5 = self.text_proj_t5(encoder_hidden_states_t5)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1).contiguous()

        # 4. Transformer blocks
        chunk_size = max(round_func(hidden_states.size(1) / self.num_chunk), 1)
        for block in self.transformer_blocks:
            chunk_cache = []
            for i in range(0, hidden_states.size(1), chunk_size):
                hs_chunk = hidden_states[:, i:i+chunk_size, :]
                enc_chunk = encoder_hidden_states  
                hs_out, enc_out = block(hs_chunk, enc_chunk, temb, image_rotary_emb)
                chunk_cache.append(hs_out)
            hidden_states = torch.cat(chunk_cache, dim=1)

        hidden_states = self.norm_final(hidden_states)

        # 5. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb=temb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, video_length, post_patch_height, post_patch_width, channels, p, p)
        output = output.permute(0, 4, 1, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

from diffusers.models.transformers.transformer_easyanimate import EasyAnimateTransformerBlock
class BatchEasyAnimateTransformerBlock(MemoryEfficientMixin, EasyAnimateTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_size = max(round_func(hidden_states.size(1) / self.num_chunk), 1)

        hs_chunks, enc_chunks = [], []
        for i in range(0, hidden_states.size(1), chunk_size):
            hs_chunk = hidden_states[:, i:i+chunk_size, :]
            enc_chunk = encoder_hidden_states  
            # -------- 1. Attention --------
            norm_hs, norm_enc, gate_msa, enc_gate_msa = self.norm1(hs_chunk, enc_chunk, temb)
            attn_hs, attn_enc = self.attn1(
                hidden_states=norm_hs,
                encoder_hidden_states=norm_enc,
                image_rotary_emb=image_rotary_emb,
            )
            hs_chunk = hs_chunk + gate_msa.unsqueeze(1) * attn_hs
            enc_chunk = enc_chunk + enc_gate_msa.unsqueeze(1) * attn_enc

            # -------- 2. Feed-forward --------
            norm_hs, norm_enc, gate_ff, enc_gate_ff = self.norm2(hs_chunk, enc_chunk, temb)
            if self.norm3 is not None:
                norm_hs = self.norm3(self.ff(norm_hs))
                if self.txt_ff is not None:
                    norm_enc = self.norm3(self.txt_ff(norm_enc))
                else:
                    norm_enc = self.norm3(self.ff(norm_enc))
            else:
                norm_hs = self.ff(norm_hs)
                if self.txt_ff is not None:
                    norm_enc = self.txt_ff(norm_enc)
                else:
                    norm_enc = self.ff(norm_enc)

            hs_chunk = hs_chunk + gate_ff.unsqueeze(1) * norm_hs
            enc_chunk = enc_chunk + enc_gate_ff.unsqueeze(1) * norm_enc

            hs_chunks.append(hs_chunk)
            enc_chunks.append(enc_chunk)

        hidden_states = torch.cat(hs_chunks, dim=1)
        encoder_hidden_states = torch.cat(enc_chunks, dim=1)

        return hidden_states, encoder_hidden_states

from diffusers.models.transformers.transformer_easyanimate import EasyAnimateLayerNormZero
class BatchEasyAnimateLayerNormZero(MemoryEfficientMixin, EasyAnimateLayerNormZero):
    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. projection 
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)

        # 2. hidden_states
        chunk_size = max(round_func(hidden_states.size(1) / self.num_chunk), 1)
        hs_chunks = []
        for i in range(0, hidden_states.size(1), chunk_size):
            hs_chunk = hidden_states[:, i:i+chunk_size, :]
            hs_chunk = self.norm(hs_chunk) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            hs_chunks.append(hs_chunk)
        hidden_states = torch.cat(hs_chunks, dim=1)

        # 3. encoder_hidden_states
        encoder_hidden_states = (
            self.norm(encoder_hidden_states) * (1 + enc_scale.unsqueeze(1)) + enc_shift.unsqueeze(1)
        )

        return hidden_states, encoder_hidden_states, gate, enc_gate