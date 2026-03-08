#copy from PCS 25


import os
import numpy as np
import torch
from tempfile import TemporaryDirectory
from plyfile import PlyElement, PlyData

# 获取项目根目录 (utils 文件夹的上级目录)
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 根据操作系统选择不同的默认可执行文件名称
_TMC3_EXE_NAME = 'tmc3.exe' if os.name == 'nt' else 'tmc3'
DEFAULT_GPCC_PATH = os.path.join(_ROOT_DIR, 'tmc13', _TMC3_EXE_NAME)


def write_ply_geo_ascii(geo_data: np.ndarray, ply_path: str) -> None:
    """
    Write geometry point cloud to a .ply file in ASCII format.
    """
    assert ply_path.endswith('.ply'), 'Destination path must be a .ply file.'
    assert geo_data.ndim == 2 and geo_data.shape[1] == 3, 'Input data must be a 3D point cloud.'
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        # write header
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n',
                      'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        # write data
        for point in geo_data:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')

def gpcc_encode(encoder_path: str, ply_path: str, bin_path: str) -> None:
    """
    Compress geometry point cloud by GPCC codec.
    """
    # Windows 不需要 wine，Linux 需要 wine
    # wine 是用于在 Linux 上运行 Windows 程序的工具，如果exe是linux上生成的，那么linux系统上也不需要wine
    wine_prefix = '' if os.name == 'nt' else 'wine '
    enc_cmd = (f'{wine_prefix}{encoder_path} '
               f'--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 '
               f'--intra_pred_max_node_size_log2=6 --positionQuantizationScale=1 --inferredDirectCodingMode=1 '
               f'--maxNumQtBtBeforeOt=4 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 '
               f'--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} ')
    enc_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'

def compress_gpcc(x: torch.Tensor, gpcc_codec_path=DEFAULT_GPCC_PATH) -> bytes:
    """
    Compress geometry point cloud by GPCC codec.
    """
    assert len(x.shape) == 2 and x.shape[1] == 3, f'Input data must be a 3D point cloud, but got {x.shape}.'

    with TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, 'point_cloud.ply')
        bin_path = os.path.join(temp_dir, 'point_cloud.bin')
        write_ply_geo_ascii(x, ply_path=ply_path)
        gpcc_encode(encoder_path=gpcc_codec_path, ply_path=ply_path, bin_path=bin_path)
        with open(bin_path, 'rb') as f:
            strings = f.read()
    return strings

def gpcc_decode(decoder_path: str, bin_path: str, recon_path: str) -> None:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    # Windows 不需要 wine，Linux 需要 wine
    wine_prefix = '' if os.name == 'nt' else 'wine '
    dec_cmd = (f'{wine_prefix}{decoder_path} '
               f'--mode=1 --outputBinaryPly=1 '
               f'--compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} ')
    dec_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'

def read_ply_geo_bin(ply_path: str) -> np.ndarray:
    """
    Read geometry point cloud from a .ply file in binary format.
    """
    assert ply_path.endswith('.ply'), 'Source path must be a .ply file.'

    ply_data = PlyData.read(ply_path).elements[0]
    means = np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)  # shape (N, 3)
    return means

def decompress_gpcc(strings: bytes, gpcc_codec_path=DEFAULT_GPCC_PATH) -> torch.Tensor:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    with TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, 'point_cloud.ply')
        bin_path = os.path.join(temp_dir, 'point_cloud.bin')
        with open(bin_path, 'wb') as f:
            f.write(strings)
        gpcc_decode(decoder_path=gpcc_codec_path, bin_path=bin_path, recon_path=ply_path)
        x = read_ply_geo_bin(ply_path=ply_path)
    return x



