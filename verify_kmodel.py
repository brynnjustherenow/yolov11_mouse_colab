import os
import sys
import csv
import math
import numpy as np
import nncase
from PIL import Image
from pathlib import Path

# NNCASE_PLUGIN_PATH 必须在 import nncase 前设置
_sp = [p for p in sys.path if p.endswith('site-packages')]
if _sp:
    os.environ.setdefault("NNCASE_PLUGIN_PATH", os.path.join(_sp[0], "nncase"))

# ==================== 配置参数（与转换脚本保持一致）====================
ONNX_PATH = "exp.onnx"
CALIB_DIR = "images"
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
TARGET = "k230"
SAMPLE_COUNT = 10
CALIB_METHOD = 'NoClip'   # 与 onnx2kmodel.py 的 PTQ_OPTION 对应
QUANT_TYPE = 'uint8'
W_QUANT_TYPE = 'uint8'
DUMP_DIR = "quant_err_dump"
THRESHOLD = 0.99          # 余弦相似度达标阈值
# ================================================================


def gen_calib_data(shape, batch, calib_dir):
    """生成 uint8 校准数据，返回 set_tensor_data 所需格式"""
    paths = sorted([p for p in Path(calib_dir).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not paths:
        raise RuntimeError(f"在 {calib_dir} 中未找到图片")
    data = []
    for i in range(batch):
        img = Image.open(paths[i % len(paths)]).convert('RGB')
        img = img.resize((shape[3], shape[2]), Image.BILINEAR)
        img = np.asarray(img, dtype=np.uint8)
        img = np.transpose(img, (2, 0, 1))
        data.append([np.ascontiguousarray(img[np.newaxis, ...])])
    return np.array(data)


def compile_with_quant_error(input_shape):
    """编译 kmodel 并 dump 逐层量化误差，返回 kmodel 字节和误差 CSV 路径"""
    co = nncase.CompileOptions()
    co.target = TARGET
    co.preprocess = True
    co.swapRB = False
    co.input_shape = input_shape
    co.input_type = 'uint8'
    co.input_range = [0, 1]
    co.mean = [0, 0, 0]
    co.std = [1, 1, 1]
    co.input_layout = "NCHW"
    co.dump_dir = DUMP_DIR

    compiler = nncase.Compiler(co)
    with open(ONNX_PATH, 'rb') as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = SAMPLE_COUNT
    ptq.calibrate_method = CALIB_METHOD
    ptq.quant_type = QUANT_TYPE
    ptq.w_quant_type = W_QUANT_TYPE
    ptq.dump_quant_error = True
    ptq.set_tensor_data(gen_calib_data(input_shape, SAMPLE_COUNT, CALIB_DIR))
    compiler.use_ptq(ptq)

    print(f"正在编译并收集量化误差 (target={TARGET}, {CALIB_METHOD}/{QUANT_TYPE}/w={W_QUANT_TYPE})...")
    compiler.compile()
    return compiler.gencode_tobytes()


def find_quant_error_csv():
    """在 dump 目录中查找 quant_error.csv"""
    for root, dirs, files in os.walk(DUMP_DIR):
        for fn in files:
            if fn == "quant_error.csv":
                return os.path.join(root, fn)
    return None


def parse_quant_error(csv_path):
    """解析 quant_error.csv，返回 [(name, cosine, mre), ...]"""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for parts in reader:
            r = dict(zip(header, [p.strip() for p in parts]))
            name = r['name']
            cos = float(r['cosine_error'])    # 实为余弦相似度 (1=完美)
            mre = float(r['mre_error'])
            rows.append((name, cos, mre))
    return rows


def main():
    if not os.path.exists(ONNX_PATH):
        print(f"找不到 {ONNX_PATH}")
        return

    iw = int(math.ceil(INPUT_WIDTH / 32.0)) * 32
    ih = int(math.ceil(INPUT_HEIGHT / 32.0)) * 32
    input_shape = [1, 3, ih, iw]

    # 编译并收集量化误差
    kmodel = compile_with_quant_error(input_shape)

    csv_path = find_quant_error_csv()
    if not csv_path:
        print("未找到 quant_error.csv，dump 失败")
        return

    rows = parse_quant_error(csv_path)

    # 区分权重和激活输出
    weight_rows = [r for r in rows if 'weight' in r[0]]
    act_rows = [r for r in rows if 'weight' not in r[0]]

    cosines = [r[1] for r in act_rows]
    mres = [r[2] for r in act_rows]

    print("=" * 60)
    print("量化误差报告 (基于 PTQ 校准阶段逐层对比 float vs quantized)")
    print("=" * 60)
    print(f"总层数: {len(rows)} (激活输出 {len(act_rows)} 层, 权重 {len(weight_rows)} 层)")
    print()
    print("【激活输出余弦相似度统计】(1.0 = 完美, 越高越好)")
    print(f"  最高: {max(cosines):.6f}")
    print(f"  最低: {min(cosines):.6f}")
    print(f"  平均: {np.mean(cosines):.6f}")
    print(f"  中位数: {np.median(cosines):.6f}")
    print()
    print("【激活输出 MRE (平均相对误差) 统计】(越低越好)")
    print(f"  最高: {max(mres):.4f}")
    print(f"  平均: {np.mean(mres):.4f}")
    print()

    # 最差的 10 层
    worst = sorted(act_rows, key=lambda x: x[1])[:10]
    print("【量化误差最大的 10 层】(重点关注)")
    for name, cos, mre in worst:
        flag = "  <== 偏低" if cos < THRESHOLD else ""
        print(f"  {cos:.6f}  MRE={mre:.4f}  {name}{flag}")
    print()

    # 达标率
    pass_count = sum(1 for c in cosines if c >= THRESHOLD)
    pass_rate = pass_count / len(cosines) * 100
    print(f"【达标率】{pass_count}/{len(cosines)} 层 >= {THRESHOLD} ({pass_rate:.1f}%)")

    # 最终输出层
    final_candidates = [r for r in act_rows if 'Sigmoid' in r[0] or 'output' in r[0].lower()]
    if final_candidates:
        final = final_candidates[-1]
        print(f"【最终输出层】{final[0]}")
        print(f"  余弦相似度: {final[1]:.6f}  MRE: {final[2]:.4f}")
    print()
    print("=" * 60)

    # 总体结论（以最终输出相似度为主要判据）
    avg_cos = np.mean(cosines)
    min_cos = min(cosines)
    final_cos = final[1] if final_candidates else avg_cos

    print("=" * 60)
    if final_cos >= THRESHOLD:
        print(f"结论：最终输出相似度 {final_cos:.4f} >= {THRESHOLD}，量化质量达标，可放心部署。")
        if min_cos < 0.98:
            print(f"  (注：少数中间层最低 {min_cos:.4f}，但误差未传导到输出，属正常)")
    elif final_cos >= 0.98:
        print(f"结论：最终输出相似度 {final_cos:.4f}，可用但有轻微损失，建议上板实测确认。")
    else:
        print(f"结论：最终输出相似度 {final_cos:.4f}，量化损失较大，建议：")
        print("  1. 把 onnx2kmodel.py 的 PTQ_OPTION 改为 4 (Kld + int16 权重)")
        print("  2. 增大 SAMPLE_COUNT 到 30~50")
        print("  3. 检查 ONNX 是否带了难以量化的后处理层")
    print()
    print(f"kmodel 已生成 ({len(kmodel) / 1024 / 1024:.2f} MB)，量化详情见 {DUMP_DIR}/")


if __name__ == '__main__':
    main()
