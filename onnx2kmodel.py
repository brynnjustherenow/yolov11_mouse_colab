import os
import csv
import argparse
import numpy as np
import nncase
import math
import onnxsim
import onnx
from PIL import Image
from pathlib import Path

# ==================== 默认配置（可通过命令行覆盖）====================
MODEL_PATH = "exp.onnx"
CALIB_DIR = "images"
OUTPUT_KMODEL = "exp.kmodel"
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
TARGET = "k230"
SAMPLE_COUNT = 10
# ================================================================

# 6 种官方量化方案
PTQ_SCHEMES = [
    ('NoClip', 'uint8', 'uint8'),   # 0: 最小最快，精度损失最大
    ('NoClip', 'uint8', 'int16'),   # 1: 权重int16，折中
    ('NoClip', 'int16', 'uint8'),   # 2: 激活int16
    ('Kld',    'uint8', 'uint8'),   # 3: Kld标定+全uint8
    ('Kld',    'uint8', 'int16'),   # 4: 推荐高精度方案
    ('Kld',    'int16', 'uint8'),   # 5: 激活int16+Kld
]


def onnx_simplify(model_file, dump_dir, input_shape):
    onnx_model = onnx.load(model_file)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_all = [node.name for node in onnx_model.graph.input]
    input_init = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_init))
    input_tensors = [n for n in onnx_model.graph.input if n.name in input_names]
    input_shapes = {}
    for node in input_tensors:
        dims = node.type.tensor_type.shape.dim
        input_shapes[node.name] = [(d.dim_value if d.dim_value != 0 else s)
                                   for d, s in zip(dims, input_shape)]
    onnx_model, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes)
    assert check, "ONNX 模型简化校验失败"
    os.makedirs(dump_dir, exist_ok=True)
    out = os.path.join(dump_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, out)
    return out


def generate_data(shape, batch, calib_dir):
    img_paths = sorted([p for p in Path(calib_dir).iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not img_paths:
        raise RuntimeError(f"在 {calib_dir} 中未找到图片")
    data = []
    for i in range(batch):
        img = Image.open(img_paths[i % len(img_paths)]).convert('RGB')
        img = img.resize((shape[3], shape[2]), Image.BILINEAR)
        img = np.asarray(img, dtype=np.uint8)
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img[np.newaxis, ...])
        data.append([img])
    return np.array(data)


def report_quant_error(dump_dir, brief=False):
    """解析 quant_error.csv，返回 (最终输出相似度, 报告文本)，brief 时只返回数值不打印"""
    csv_path = None
    for root, dirs, files in os.walk(dump_dir):
        for fn in files:
            if fn == "quant_error.csv":
                csv_path = os.path.join(root, fn)
                break
    if not csv_path:
        return None

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = []
        for parts in reader:
            r = dict(zip(header, [p.strip() for p in parts]))
            rows.append((r['name'], float(r['cosine_error']), float(r['mre_error'])))

    act_rows = [r for r in rows if 'weight' not in r[0]]
    cosines = [r[1] for r in act_rows]
    mres = [r[2] for r in act_rows]
    final_candidates = [r for r in act_rows if 'Sigmoid' in r[0] or 'output' in r[0].lower()]
    final = final_candidates[-1] if final_candidates else (None, float(np.mean(cosines)), None)
    pass_count = sum(1 for c in cosines if c >= 0.99)

    if brief:
        return final[1]

    threshold = 0.99
    print("\n" + "=" * 55)
    print("量化质量报告 (PTQ float vs quantized 逐层对比)")
    print("=" * 55)
    print(f"激活输出层: {len(act_rows)} 层")
    print(f"  余弦相似度  平均={np.mean(cosines):.6f}  最低={min(cosines):.6f}")
    print(f"  MRE         平均={np.mean(mres):.4f}    最高={max(mres):.4f}")
    print(f"  达标率(>=0.99): {pass_count}/{len(cosines)} ({pass_count/len(cosines)*100:.1f}%)")
    worst = sorted(act_rows, key=lambda x: x[1])[:5]
    print("\n  误差最大的 5 层:")
    for name, cos, mre in worst:
        print(f"    {cos:.6f}  MRE={mre:.4f}  {name}")
    print(f"\n  最终输出层: {final[0]}")
    print(f"    余弦相似度: {final[1]:.6f}")
    print("-" * 55)
    if final[1] >= threshold:
        print(f"结论: 最终输出 {final[1]:.4f} >= {threshold}，量化质量达标，可放心部署。")
        if min(cosines) < 0.98:
            print(f"  (少数中间层最低 {min(cosines):.4f}，但误差未传导到输出，属正常)")
    elif final[1] >= 0.98:
        print(f"结论: 最终输出 {final[1]:.4f}，可用但有轻微损失，建议上板实测确认。")
    else:
        print(f"结论: 最终输出 {final[1]:.4f}，量化损失较大，建议换更高精度方案。")
    print("=" * 55)
    return final[1]


def compile_once(input_shape, model_content, calib_data, ptq_option, output_name,
                 target, samples, dump_dir):
    """编译单个 PTQ 方案，保存 kmodel，打印质量报告，返回 (最终相似度, 文件大小MB)"""
    calib_method, quant_type, w_quant_type = PTQ_SCHEMES[ptq_option]
    tag = f"方案{ptq_option}: {calib_method}/{quant_type}/w={w_quant_type}"
    print(f"\n{'='*55}")
    print(f"编译 {tag}")
    print(f"{'='*55}")

    co = nncase.CompileOptions()
    co.target = target
    co.preprocess = True
    co.swapRB = False
    co.input_shape = input_shape
    co.input_type = 'uint8'
    co.input_range = [0, 1]
    co.mean = [0, 0, 0]
    co.std = [1, 1, 1]
    co.input_layout = "NCHW"
    co.dump_dir = dump_dir

    compiler = nncase.Compiler(co)
    compiler.import_onnx(model_content, nncase.ImportOptions())

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = samples
    ptq.calibrate_method = calib_method
    ptq.quant_type = quant_type
    ptq.w_quant_type = w_quant_type
    ptq.dump_quant_error = True
    ptq.set_tensor_data(calib_data)
    compiler.use_ptq(ptq)

    print("正在编译...")
    compiler.compile()

    kmodel = compiler.gencode_tobytes()
    with open(output_name, 'wb') as f:
        f.write(kmodel)
    size_mb = len(kmodel) / 1024 / 1024
    print(f"已保存: {output_name} ({size_mb:.2f} MB)")

    final_cos = report_quant_error(dump_dir)
    return final_cos, size_mb


def main():
    parser = argparse.ArgumentParser(
        prog="onnx2kmodel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="K230 ONNX → kmodel 转换工具 (nncase 2.9.0)",
        epilog="""
量化方案说明:
  0  NoClip/uint8/uint8    最小最快，精度损失最大
  1  NoClip/uint8/int16    权重int16，体积速度折中
  2  NoClip/int16/uint8    激活int16
  3  Kld/uint8/uint8       Kld标定，检测任务常用
  4  Kld/uint8/int16       推荐高精度方案
  5  Kld/int16/uint8       激活int16+Kld

示例:
  python onnx2kmodel.py              默认方案(0)
  python onnx2kmodel.py -p 4         高精度方案(4)
  python onnx2kmodel.py -a           转换全部6种方案
""")
    parser.add_argument('-p', '--ptq', type=int, default=None, metavar='N',
                        choices=range(6),
                        help="量化方案编号 0-5 (默认 0)")
    parser.add_argument('-a', '--all', action='store_true',
                        help="转换全部 6 种量化方案，分别输出")
    parser.add_argument('--model', default=MODEL_PATH, metavar='PATH',
                        help=f"ONNX 模型路径 (默认: {MODEL_PATH})")
    parser.add_argument('--calib', default=CALIB_DIR, metavar='DIR',
                        help=f"校准图片目录 (默认: {CALIB_DIR})")
    parser.add_argument('--target', default=TARGET, metavar='T',
                        help=f"目标平台 (默认: {TARGET})")
    parser.add_argument('--samples', type=int, default=SAMPLE_COUNT, metavar='N',
                        help=f"校准样本数 (默认: {SAMPLE_COUNT})")
    parser.add_argument('-o', '--output', default=OUTPUT_KMODEL, metavar='PATH',
                        help=f"输出 kmodel 文件名 (默认: {OUTPUT_KMODEL}，-a 模式忽略)")
    args = parser.parse_args()

    print("=" * 55)
    print("K230 模型转换工具 (nncase)")
    print("=" * 55)

    if not os.path.exists(args.model):
        print(f"找不到 ONNX 模型: {args.model}")
        return
    if not os.path.exists(args.calib):
        print(f"找不到校准图片目录: {args.calib}")
        return

    iw = int(math.ceil(INPUT_WIDTH / 32.0)) * 32
    ih = int(math.ceil(INPUT_HEIGHT / 32.0)) * 32
    input_shape = [1, 3, ih, iw]
    print(f"模型输入尺寸: {input_shape}")
    print(f"目标平台: {args.target}  校准样本: {args.samples}")

    # ONNX 简化（只做一次）
    dump_dir = 'tmp'
    print("简化 ONNX 模型...")
    model_file = onnx_simplify(args.model, dump_dir, input_shape)
    with open(model_file, 'rb') as f:
        model_content = f.read()

    # 生成校准数据（只做一次）
    print("生成校准数据...")
    calib_data = generate_data(input_shape, args.samples, args.calib)

    # 编译
    if args.all:
        results = []
        for i in range(6):
            base = os.path.splitext(os.path.basename(args.model))[0]
            out = f"{base}_ptq{i}.kmodel"
            dump_sub = os.path.join(dump_dir, f"ptq{i}")
            final_cos, size_mb = compile_once(
                input_shape, model_content, calib_data, i, out,
                args.target, args.samples, dump_sub)
            results.append((i, PTQ_SCHEMES[i], final_cos, size_mb, out))

        # 汇总对比
        print(f"\n{'='*70}")
        print("全部方案汇总对比")
        print(f"{'='*70}")
        print(f"{'方案':<4} {'标定/激活/权重':<22} {'最终相似度':<12} {'大小MB':<8} {'文件'}")
        print("-" * 70)
        for i, scheme, cos, size, out in results:
            cos_str = f"{cos:.6f}" if cos else "N/A"
            print(f"{i:<4} {scheme[0]+'/'+scheme[1]+'/'+scheme[2]:<22} {cos_str:<12} {size:<8.2f} {out}")
        print("-" * 70)
        best = max(results, key=lambda x: x[2] or 0)
        print(f"精度最高: 方案{best[0]} ({best[2]:.6f}) → {best[4]}")
        print("=" * 70)
    else:
        ptq_opt = args.ptq if args.ptq is not None else 0
        compile_once(input_shape, model_content, calib_data, ptq_opt,
                     args.output, args.target, args.samples, dump_dir)


if __name__ == '__main__':
    main()
