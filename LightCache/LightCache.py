from diffusers import StableVideoDiffusionPipeline, AnimateDiffPipeline, TextToVideoSDPipeline, EasyAnimatePipeline
from .memory import MemoryEfficientMixin

class LightCacher(object):
    def __init__(self, pipe=None, num_frames=25):
        if pipe is not None: self.pipe = pipe
        self.num_timesteps = len(self.pipe.scheduler.timesteps)
        # self.num_timesteps = 25
        self.num_frames = num_frames
    def enable(self, Swap=True, Slice=True, Chunk=False):
        assert self.pipe is not None
        self.reset_states()
        self.wrap_modules()
        if Chunk:
            self.warp_forward(num_temporal_chunk=9, num_spatial_chunk=2, num_frames=self.num_frames)
        if Slice:
            self.pipe.vae.use_slicing = True
        if Swap:
            self.pipe.enable_model_cpu_offload()
    def enable_chunk(self):
        assert self.pipe is not None
        self.warp_forward(num_temporal_chunk=9, num_spatial_chunk=2, num_frames=self.num_frames)
    def disable(self):
        self.unwrap_modules()
        self.reset_states()
    
    def set_params(self,cache_interval=1, cache_branch_id=0, skip_mode='uniform'):
        cache_layer_id = cache_branch_id % 3
        cache_block_id = cache_branch_id // 3
        self.params = {
            'cache_interval': cache_interval,
            'cache_layer_id': cache_layer_id,
            'cache_block_id': cache_block_id,
            'skip_mode': skip_mode
        }

    def is_skip_step(self, block_i, layer_i, blocktype = "down"):
        self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep # For some pipeline that the first timestep != 0
        cache_interval, cache_layer_id, cache_block_id, skip_mode = \
            self.params['cache_interval'], self.params['cache_layer_id'], self.params['cache_block_id'], self.params['skip_mode']
        if skip_mode == 'uniform':
            if (self.cur_timestep-self.start_timestep) % cache_interval == 0: return False
        if block_i > cache_block_id or blocktype == 'mid':
            return True
        if block_i < cache_block_id: return False
        return layer_i >= cache_layer_id if blocktype == 'down' else layer_i > cache_layer_id
        
    def is_enter_position(self, block_i, layer_i):
        return block_i == self.params['cache_block_id'] and layer_i == self.params['cache_layer_id']

    def wrap_unet_forward(self):
        self.function_dict['unet_forward'] = self.pipe.unet.forward
        def wrapped_forward(*args, **kwargs):
            self.cur_timestep = list(self.pipe.scheduler.timesteps).index(args[1].item())
            result = self.function_dict['unet_forward'](*args, **kwargs)
            return result
        self.pipe.unet.forward = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype = "down"):
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward
        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            result = self.cached_output[(blocktype, block_name, block_i, layer_i)] if skip else self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
            if not skip: self.cached_output[(blocktype, block_name, block_i, layer_i)] = result
            return result
        block.forward = wrapped_forward

    def wrap_former_forward(self):
        self.function_dict['dit_forward'] = self.pipe.transformer.forward
        def wrapped_forward(*args, **kwargs):
            t = args[1]
            import torch
            if isinstance(t, torch.Tensor):
                if t.ndim > 0:
                    t = t[0]
                t = int(t.detach().cpu())
            self.cur_timestep = t
    
            result = self.function_dict['dit_forward'](*args, **kwargs)
            return result
        self.pipe.transformer.forward = wrapped_forward
        
    def wrap_modules(self):
        if self.is_easyanimate_pipe(self.pipe):
            # 1. wrap transformer forward
            self.wrap_former_forward()
            # 2. wrap transformer_blocks forward
            for block_i, block in enumerate(self.pipe.transformer.transformer_blocks):
                if hasattr(block, "norm1"):
                    self.wrap_block_forward(block.norm1, "norm1", block_i, 0)
                if hasattr(block, "attn1"):
                    self.wrap_block_forward(block.attn1, "attn1", block_i, 0)
                if hasattr(block, "norm2"):
                    self.wrap_block_forward(block.norm2, "norm2", block_i, 0)
                if hasattr(block, "ff"):
                    self.wrap_block_forward(block.ff, "ff", block_i, 0)
                # if hasattr(block, "txt_ff"):
                #     self.wrap_block_forward(block.txt_ff, "txt_ff", block_i, 0)
                    
                self.wrap_block_forward(block, "block", block_i, 0, blocktype="down")
        else:
            # 1. wrap unet forward
            self.wrap_unet_forward()
            # 2. wrap downblock forward
            for block_i, block in enumerate(self.pipe.unet.down_blocks):
                for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                    self.wrap_block_forward(attention, "attentions", block_i, layer_i)
                for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                    self.wrap_block_forward(resnet, "resnet", block_i, layer_i)
                for downsampler in getattr(block, "downsamplers", []) if block.downsamplers else []:
                    self.wrap_block_forward(downsampler, "downsampler", block_i, len(getattr(block, "resnets", [])))
                self.wrap_block_forward(block, "block", block_i, 0, blocktype = "down")
            # 3. wrap midblock forward
            self.wrap_block_forward(self.pipe.unet.mid_block, "mid_block", 0, 0, blocktype = "mid")
            # 4. wrap upblock forward
            block_num = len(self.pipe.unet.up_blocks)
            for block_i, block in enumerate(self.pipe.unet.up_blocks):
                layer_num = len(getattr(block, "resnets", []))
                for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                    self.wrap_block_forward(attention, "attentions", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
                for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                    self.wrap_block_forward(resnet, "resnet", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
                for upsampler in getattr(block, "upsamplers", []) if block.upsamplers else []:
                    self.wrap_block_forward(upsampler, "upsampler", block_num - block_i - 1, 0, blocktype = "up")
                self.wrap_block_forward(block, "block", block_num - block_i - 1, 0, blocktype = "up")

    def unwrap_modules(self):
        # 1. unet forward
        self.pipe.unet.forward = self.function_dict['unet_forward']
        # 2. downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("down", "attentions", block_i, layer_i)]
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[("down", "resnet", block_i, layer_i)]
            for downsampler in getattr(block, "downsamplers", []) if block.downsamplers else []:
                downsampler.forward = self.function_dict[("down", "downsampler", block_i, len(getattr(block, "resnets", [])))]
            block.forward = self.function_dict[("down", "block", block_i, 0)]
        # 3. midblock forward
        self.pipe.unet.mid_block.forward = self.function_dict[("mid", "mid_block", 0, 0)]
        # 4. upblock forward
        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("up", "attentions", block_num - block_i - 1, layer_num - layer_i - 1)]
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[("up", "resnet", block_num - block_i - 1, layer_num - layer_i - 1)]
            for upsampler in getattr(block, "upsamplers", []) if block.upsamplers else []:
                upsampler.forward = self.function_dict[("up", "upsampler", block_num - block_i - 1, 0)]
            block.forward = self.function_dict[("up", "block", block_num - block_i - 1, 0)]

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None

    def is_svd_pipe(self, pipe):
        return isinstance(pipe, StableVideoDiffusionPipeline)

    def is_animate_pipe(self, pipe):
        return isinstance(pipe, AnimateDiffPipeline)
        
    def is_t2vsd_pipe(self, pipe):
        return isinstance(pipe, TextToVideoSDPipeline)

    def is_easyanimate_pipe(self, pipe):
        return isinstance(pipe, EasyAnimatePipeline)
        
    def _assign_property(self, m: MemoryEfficientMixin):
        m.num_chunk = self.num_temporal_chunk
        if self.num_spatial_chunk is not None:
            m.num_spatial_chunk = self.num_spatial_chunk

    def warp_forward(self, num_temporal_chunk, num_spatial_chunk=None, num_frames=None):
        self.num_temporal_chunk = num_temporal_chunk
        self.num_spatial_chunk = num_spatial_chunk
        
        if self.is_svd_pipe(self.pipe):
            from diffusers.models.resnet import SpatioTemporalResBlock
            from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
            from diffusers.models.attention import TemporalBasicTransformerBlock
            from diffusers.models.unets.unet_3d_blocks import (DownBlockSpatioTemporal,
                                                               CrossAttnDownBlockSpatioTemporal,
                                                               UpBlockSpatioTemporal,
                                                               CrossAttnUpBlockSpatioTemporal)
            from diffusers.models.unets import UNetSpatioTemporalConditionModel
            from .batch import BatchSpatioTemporalResBlock, BatchTransformerSpatioTemporalModel, BatchTemporalBasicTransformerBlock, BatchUNetSpatioTemporalConditionModel, BatchDownBlockSpatioTemporal, BatchCrossAttnDownBlockSpatioTemporal, BatchUpBlockSpatioTemporal, BatchCrossAttnUpBlockSpatioTemporal
            
            BatchStableVideoDiffusion = {
                UNetSpatioTemporalConditionModel: BatchUNetSpatioTemporalConditionModel,
                SpatioTemporalResBlock: BatchSpatioTemporalResBlock,  
                TransformerSpatioTemporalModel: BatchTransformerSpatioTemporalModel,
                TemporalBasicTransformerBlock: BatchTemporalBasicTransformerBlock,
                DownBlockSpatioTemporal: BatchDownBlockSpatioTemporal,
                CrossAttnDownBlockSpatioTemporal: BatchCrossAttnDownBlockSpatioTemporal,
                UpBlockSpatioTemporal: BatchUpBlockSpatioTemporal,
                CrossAttnUpBlockSpatioTemporal: BatchCrossAttnUpBlockSpatioTemporal,
            }
        
            for n, m in self.pipe.unet.named_modules():
                if isinstance(m, tuple(BatchStableVideoDiffusion.keys())):
                    m.__class__ = BatchStableVideoDiffusion[m.__class__]
                    self._assign_property(m)
            for n, m in self.pipe.vae.named_modules():
                if isinstance(m, tuple(BatchStableVideoDiffusion.keys())):
                    m.__class__ = BatchStableVideoDiffusion[m.__class__]
                    self._assign_property(m)
            
        elif self.is_animate_pipe(self.pipe):
            from diffusers.models.unets.unet_motion_model import (CrossAttnDownBlockMotion, DownBlockMotion,
                                                                  CrossAttnUpBlockMotion, UpBlockMotion,
                                                                  UNetMidBlockCrossAttnMotion)
            from .batch import BatchCrossAttnDownBlockMotion, BatchDownBlockMotion, BatchCrossAttnUpBlockMotion, BatchUpBlockMotion, BatchDownBlockSpatioTemporal, BatchUNetMidBlockCrossAttnMotion
            
            AnimateDiffScheme = {
                CrossAttnDownBlockMotion: BatchCrossAttnDownBlockMotion,
                DownBlockMotion: BatchDownBlockMotion,
                CrossAttnUpBlockMotion: BatchCrossAttnUpBlockMotion,
                UpBlockMotion: BatchUpBlockMotion,
                UNetMidBlockCrossAttnMotion: BatchUNetMidBlockCrossAttnMotion
            }
            for n, m in self.pipe.unet.named_modules():
                if isinstance(m, tuple(AnimateDiffScheme.keys())):
                    m.__class__ = AnimateDiffScheme[m.__class__]
                    self._assign_property(m)
            # self.pipe.vae.use_slicing = True

        elif self.is_t2vsd_pipe(self.pipe):
            from diffusers.models import UNet3DConditionModel
            from diffusers.models.transformers.transformer_2d import Transformer2DModel
            from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
            from diffusers.models.unets.unet_3d_blocks import CrossAttnDownBlock3D, CrossAttnUpBlock3D, DownBlock3D, UpBlock3D
            from .batch import BatchUNet3DConditionModel, BatchCrossAttnDownBlock3D, BatchDownBlock3D, BatchUpBlock3D, BatchCrossAttnUpBlock3D, BatchTransformerTemporalModel, BatchTransformer2DModel

            TextToVideoSDPipelineScheme = {
                UNet3DConditionModel: BatchUNet3DConditionModel,
                Transformer2DModel: BatchTransformer2DModel,
                TransformerTemporalModel: BatchTransformerTemporalModel,
                CrossAttnDownBlock3D: BatchCrossAttnDownBlock3D,
                DownBlock3D: BatchDownBlock3D,
                UpBlock3D: BatchUpBlock3D,
                CrossAttnUpBlock3D: BatchCrossAttnUpBlock3D
            }
            for n, m in self.pipe.unet.named_modules():
                if isinstance(m, tuple(TextToVideoSDPipelineScheme.keys())):
                    m.__class__ = TextToVideoSDPipelineScheme[m.__class__]
                    self._assign_property(m)
            for n, m in self.pipe.vae.named_modules():
                if isinstance(m, tuple(TextToVideoSDPipelineScheme.keys())):
                    m.__class__ = TextToVideoSDPipelineScheme[m.__class__]
                    self._assign_property(m)
            # self.pipe.vae.use_slicing = True

        elif self.is_easyanimate_pipe(self.pipe):
            from diffusers.models.transformers.transformer_easyanimate import EasyAnimateTransformer3DModel, EasyAnimateTransformerBlock, EasyAnimateLayerNormZero
            from .batch import BatchEasyAnimateTransformer3DModel, BatchEasyAnimateTransformerBlock, BatchEasyAnimateLayerNormZero
            EasyAnimatePipelineScheme = {
                EasyAnimateTransformer3DModel: BatchEasyAnimateTransformer3DModel,
                EasyAnimateTransformerBlock: BatchEasyAnimateTransformerBlock,
                EasyAnimateLayerNormZero: BatchEasyAnimateLayerNormZero
            }
            
            for n, m in self.pipe.transformer.named_modules():
                if isinstance(m, tuple(EasyAnimatePipelineScheme.keys())):
                    m.__class__ = EasyAnimatePipelineScheme[m.__class__]
                    self._assign_property(m)

    