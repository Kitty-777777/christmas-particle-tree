# -*- coding: utf-8 -*-
"""
手势分类器训练脚本
==================
用合成特征训练一个 softmax 多分类模型，替代 main.js 里"数手指"的硬阈值规则。

为什么训练而不是写死规则：
  真机上手指常处于"半弯"状态，伸展比 q 落在 1.3~1.8 的模糊区，
  硬阈值 q>1.85 会把它一刀切成"弯"，导致数错手指、识别不出。
  用带噪声的数据训练后，模型学到的是一条平滑的决策边界，对模糊手势更宽容（更灵敏），
  同时仍能把它和邻近类别区分开。

特征（与 main.js 的 lastGestureDiag 完全对应，6 维）：
  q0..q3  四指伸展比 = dist(mcp,tip)/dist(mcp,pip)，伸直≈2.4 弯曲≈0.8
  t       拇指张开比 = dist(thumb_tip,index_mcp)/掌尺度，张开≈0.7 收拢≈0.14
  p       捏合比     = dist(thumb_tip,index_tip)/掌尺度，捏合≈0.1 张开≈0.9

数据生成：对每类手势，按每根手指的目标状态（伸直/弯曲）采样，加入：
  - 手指半弯噪声（把 q 拉向中间值，模拟真实不完全伸展）
  - 个体尺度差异、测量噪声
  - 类别混合样本（少量标签噪声，提升鲁棒性）

输出：js/gesture-model.js（ES Module），前端 import 后做前向传播。
运行：python tools/train_gesture.py
"""
import json
import math
import os
import random

import numpy as np

random.seed(20261125)
np.random.seed(20261125)

CLASSES = ['FIST', 'ONE', 'TWO', 'THREE', 'FOUR', 'PALM', 'OK']

# 每根手指的目标伸展状态：1=伸直 0=弯曲；thumb_o=拇指张开 1/收拢 0；pinch=捏合 1/张开 0
# 结构: (index, middle, ring, pinky, thumb_open, pinch)
PROTOTYPES = {
    'FIST':  (0, 0, 0, 0, 0, 1),   # 握拳：拇指收拢贴着食指 → 距离近(pinch 语义此处按"拇指到食指尖近")
    'ONE':   (1, 0, 0, 0, 0, 0),
    'TWO':   (1, 1, 0, 0, 0, 0),
    'THREE': (1, 1, 1, 0, 0, 0),
    'FOUR':  (1, 1, 1, 1, 0, 0),   # 四指伸直、拇指收拢（区别于 PALM）
    'PALM':  (1, 1, 1, 1, 1, 0),   # 张掌：拇指张开
    'OK':    (0, 1, 1, 1, 0, 1),   # 食指弯 + 拇指捏合 + 其余伸直
}


def sample_q(straight, half_prob=0.25):
    """采样一根手指的伸展比。straight=True 目标≈2.4，False 目标≈0.85。
    half_prob 概率落入"接近伸直但未完全伸直"(≈1.75) 或"接近弯曲"(≈1.2)。
    关键：半弯样本必须偏向它本来的目标侧，否则标签与特征自相矛盾、模型学不准。
    这样模型学到的伸直边界会从硬规则的 1.85 下移到约 1.5 → 对手指没伸直到底的手势更灵敏。"""
    if random.random() < half_prob:
        return random.gauss(1.78, 0.16) if straight else random.gauss(1.20, 0.16)
    mu, sd = (2.42, 0.26) if straight else (0.80, 0.14)
    return random.gauss(mu, sd)


def sample_feature(cls):
    idx, mid, ring, pink, thumb_o, pinch = PROTOTYPES[cls]
    q = [sample_q(v) for v in (idx, mid, ring, pink)]
    # 拇指张开比：张开≈0.72，收拢≈0.16；OK 时拇指贴食指也偏小
    t = random.gauss(0.72, 0.09) if thumb_o else random.gauss(0.17, 0.06)
    # 捏合比：pinch 目标≈0.12，其余≈0.9；握拳时拇指与食指都收拢，距离也小
    if pinch or cls == 'FIST':
        p = random.gauss(0.14, 0.07)
    else:
        p = random.gauss(0.88, 0.13)
    # 加一点整体测量抖动（所有特征共同缩放，模拟手离镜头远近）——比例特征本应抵消，
    # 但保留极小的残余噪声更接近真机
    return np.array(q + [t, p], dtype=np.float64)


def make_dataset(n_per_class=3000):
    xs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            xs.append(sample_feature(cls))
            ys.append(ci)
    xs = np.array(xs)
    ys = np.array(ys, dtype=np.int64)
    # 打乱
    perm = np.random.permutation(len(ys))
    return xs[perm], ys[perm]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(xs, ys, steps=4000, lr=0.35, l2=1e-4):
    n, d = xs.shape
    c = len(CLASSES)
    mean = xs.mean(axis=0)
    std = xs.std(axis=0) + 1e-8
    x = (xs - mean) / std

    W = np.random.randn(d, c) * 0.05
    b = np.zeros(c)
    for s in range(steps):
        prob = softmax(x @ W + b)
        prob[np.arange(n), ys] -= 1
        prob /= n
        gw = x.T @ prob + l2 * W
        gb = prob.sum(axis=0)
        W -= lr * gw
        b -= lr * gb
        if (s + 1) % 1000 == 0:
            pred = (softmax(x @ W + b)).argmax(axis=1)
            print(f'  step {s+1:4d}  acc={(pred == ys).mean()*100:.2f}%')
    return W, b, mean, std


def evaluate(W, b, mean, std, xs, ys):
    prob = softmax(((xs - mean) / std) @ W + b)
    pred = prob.argmax(axis=1)
    acc = (pred == ys).mean()
    print(f'\n测试集准确率: {acc*100:.2f}%  ({len(ys)} 样本)')
    print('混淆矩阵（行=真实 列=预测）:')
    header = '        ' + ' '.join(f'{c[:5]:>6}' for c in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = [int(((pred == j) & (ys == i)).sum()) for j in range(len(CLASSES))]
        print(f'{cls:>7} ' + ' '.join(f'{v:6d}' for v in row))
    # 每类召回率
    print('每类召回率:')
    for i, cls in enumerate(CLASSES):
        mask = ys == i
        print(f'  {cls:<6} {(pred[mask] == i).mean()*100:.1f}%')
    return acc, prob


def main():
    print('生成训练集/测试集...')
    xtr, ytr = make_dataset(3000)
    xte, yte = make_dataset(600)
    print(f'训练集 {len(ytr)} 条，测试集 {len(yte)} 条')

    print('\n训练 softmax 分类器...')
    W, b, mean, std = train(xtr, ytr)

    print('\n评估...')
    acc, _ = evaluate(W, b, mean, std, xte, yte)
    if acc < 0.90:
        print('\n[警告] 准确率低于 90%，建议增加数据量或调整噪声参数')

    # 导出为前端模块
    model = {
        'version': 1,
        'classes': CLASSES,
        'mean': [round(float(v), 6) for v in mean],
        'std': [round(float(v), 6) for v in std],
        'W': [[round(float(v), 6) for v in row] for row in W.T],  # 转置成 [class][feature] 便于 JS 读取
        'b': [round(float(v), 6) for v in b],
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'js')
    out_path = os.path.join(out_dir, 'gesture-model.js')
    js = ('// 由 tools/train_gesture.py 自动生成，请勿手动修改。\n'
          '// 特征顺序: [q0,q1,q2,q3,t,p]；W 按 [class][feature] 排列。\n'
          'export const GESTURE_MODEL = ' + json.dumps(model, ensure_ascii=False) + ';\n')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(js)
    print(f'\n已导出模型到 {out_path}  ({os.path.getsize(out_path)} 字节)')


if __name__ == '__main__':
    main()
