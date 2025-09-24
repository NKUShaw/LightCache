import torch
import numpy as np

def compress_exponent_blocks(exponent_bits, block_size):
    bitstream = []
    header = []

    for i in range(0, len(exponent_bits), block_size):
        block = exponent_bits[i:i + block_size]
        min_exp = block.min()
        max_exp = block.max()
        bit_width = int(np.ceil(np.log2(max_exp - min_exp + 1)))
        header.append((min_exp, bit_width))

        relative_exp = block - min_exp
        for val in relative_exp:
            bitstream.append(format(int(val), f'0{bit_width}b'))  # ✅ 最终版修复

    return bitstream, header


def decompress_exponent_blocks(bitstream, header, total_len, block_size=1024):
    decoded = []
    idx = 0
    for i, (min_exp, bit_width) in enumerate(header):
        block_len = min(block_size, total_len - i * block_size)
        for _ in range(block_len):
            val = int(bitstream[idx], 2) + min_exp
            decoded.append(val)
            idx += 1
    return np.array(decoded, dtype=np.uint16)

def compress_model_exponent(unet, block_size=1024):
    exponent_headers = {}
    exponent_streams = {}

    for name, param in unet.named_parameters():
        if not param.dtype == torch.float16:
            continue
        fp16_tensor = param.detach().cpu().numpy()
        bits = fp16_tensor.view(np.uint16)
        exponent_bits = (bits >> 10) & 0x1F

        # 压缩
        stream, header = compress_exponent_blocks(exponent_bits, block_size)
        exponent_headers[name] = header
        exponent_streams[name] = stream

    return exponent_streams, exponent_headers



def decompress_model_exponent(unet, exponent_streams, exponent_headers, original_unet):
    for name, param in unet.named_parameters():
        if name not in exponent_streams:
            continue
        if not param.dtype == torch.float16:
            continue

        original_tensor = original_unet.state_dict()[name].detach().cpu().numpy()
        fp16_bits = original_tensor.view(np.uint16)

        sign_bits = (fp16_bits >> 15) & 0x1
        mantissa_bits = fp16_bits & 0x3FF

        # 还原 exponent
        decoded_exponent = decompress_exponent_blocks(
            exponent_streams[name], exponent_headers[name], total_len=len(fp16_bits)
        )
        reconstructed_bits = (sign_bits << 15) | (decoded_exponent << 10) | mantissa_bits
        reconstructed_fp16 = reconstructed_bits.view(np.float16)

        # 替换原参数
        param.data.copy_(torch.from_numpy(reconstructed_fp16).view_as(param.data).to(param.data.device))
