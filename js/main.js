/* ============================================================================
 * 3D 粒子圣诞树 · 手势交互
 * ----------------------------------------------------------------------------
 * 技术栈：Three.js (ES Module) + MediaPipe HandLandmarker (WebGL/WASM) + GLSL ShaderMaterial
 * 结构：
 *   1. 配置区（文字 / 粒子数 / 树形参数都在这里改）
 *   2. DOM 引用
 *   3. 场景 / 相机 / 渲染器
 *   4. 工具：径向渐变纹理
 *   5. 星空背景粒子
 *   6. 文字采样 → 粒子目标点
 *   7. 圣诞树粒子系统（核心：树 ↔ 文字的 GPU 插值）
 *   8. 树顶 / 地面光晕
 *   9. 拖尾后处理
 *  10. 状态机与效果映射
 *  11. 手势识别（MediaPipe）
 *  12. 动画循环 / 事件
 * ========================================================================== */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { FilesetResolver, HandLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';
import { GESTURE_MODEL } from './gesture-model.js';

/* ========================= 1. 配置区 =========================
 * 想改文字、粒子数量、树的大小，只需要动这里。
 */
const CONFIG = {
  /* ---------------- 粒子系统 ---------------- */
  particleCount: window.innerWidth < 768 ? 4000 : 9000, // 主粒子数（树/文字/星球/相框共用）
  starCount: window.innerWidth < 768 ? 500 : 1400,      // 背景星空粒子数
  confettiCount: window.innerWidth < 768 ? 700 : 1600,  // 相册彩屑粒子数
  particleSize: 4.8,      // 粒子基础尺寸（个体大小总开关，越大光点越饱满）
  glowStrength: 1.0,      // 主粒子发光强度增益（1.0=默认，调大更亮）

  /* ---------------- 圣诞树几何 ---------------- */
  tree: {
    height: 12,     // 树高（世界单位）
    radius: 4.4,    // 底部半径
    arms: 5,        // 螺旋臂数量
    turns: 6.2,     // 每条臂绕轴圈数
    colors: ['#ff5fa8', '#ffa8d4', '#ffd9ec'], // 底部→中部→顶部的粉色渐变
  },

  /* ---------------- 粒子文字 ---------------- */
  text: {
    first: '张嘉雯圣诞快乐',   // OK 手势
    second: '今天你开心了吗？', // 张开手掌
    size: 14,       // 文字世界宽度（越大字越大）
    widthRatio: 0.82, // 文字最多占可见宽度比例（窄屏自适应不裁切）
  },

  /* ---------------- 手势识别（新版 MediaPipe HandLandmarker · v29 训练模型）----------------
   * HandLandmarker 走 WebGL/WASM，不依赖 WebGPU，返回标准 21 点（含 z 深度）。
   * 分类用 tools/train_gesture.py 训练的 softmax 模型（gesture-model.js，测试准确率 99%），
   * 对"手指半弯"的宽容度远高于旧的硬阈值数手指规则 → 更灵敏。
   * 灵敏度相关参数：minConf=判定阈值(调低更灵敏) emaAlpha=置信度平滑(调高更快) triggerGap=同手势重复触发间隔。 */
  gesture: {
    minHandScore: 0.3,     // 引擎层手部检测置信度（0.3 更容易检出；调高减少误检）
    zWeight: 0.8,          // z 在距离里的权重（深度噪声较大，降权）
    delegate: 'GPU',       // HandLandmarker 推理后端：GPU=WebGL（优先）/ CPU=WASM（兜底）
    minConf: 0.55,         // 模型概率门槛：最高类概率低于此值视为"识别中"（调低更灵敏、更易误触）
    emaAlpha: 0.45,        // 置信度指数平滑系数（0~1，越大越快跟手，越小越稳）
    triggerFrames: 2,      // 平滑置信度连续超阈值的帧数才触发（旧版硬等 3 帧，这里更快）
    triggerGap: 380,       // 同一手势两次触发最小间隔 ms（旧版 500，调短更灵敏）
    // 下面三个是旧规则法的参数，保留仅用于兜底（模型加载失败时）
    straightRatio: 1.85,
    thumbSpread: 0.55,
    okRatio: 0.42,
  },

  /* ---------------- 礼物相册照片 ----------------
   * ★ 替换照片就改这里 ★
   *   - 本地图片：src: './photos/1.jpg'（把图片放到项目目录）
   *   - 网络图片：src: 'https://example.com/photo.jpg'
   *   - 用户上传：把上传得到的可访问 URL 填进 src
   * src 留空 → 自动生成渐变占位纹理；加载失败 → console.warn 并回退占位，绝不崩溃。
   * 光球数量随数组长度自动生成，一一对应，无需手动改动。
   * 照片以竖版比例贴在中央相框内。
   */
  photos: [
    { src: '', title: '圣诞快乐 2026' },
    { src: '', title: '愿所有美好如约而至' },
    { src: '', title: 'Merry & Bright' },
    { src: '', title: '把魔法带回身边' },
    { src: '', title: '星光与祝福' },
    { src: '', title: 'Winter Wonderland' },
    { src: '', title: '把快乐装进礼物盒' },
    { src: '', title: '新的篇章' },
  ],

  /* ---------------- 相册彩屑 / 占位图配色 ---------------- */
  confettiColors: ['#b06bff', '#4db8ff', '#5fe0a8', '#ffd45c', '#ff9ec7', '#8ab6ff'],

  /* ---------------- 光球（粒子里的一颗"大粒子"）----------------
   * 视觉语言与主粒子系统一致：加法混合的柔和径向光 + 极淡浅蓝白边缘光，
   * 中心封着一张若隐若现的圆形照片。平时像树上恰好有颗更大的光点，不突兀；
   * 进入相册后照片变清晰、边缘光与光晕增强。所有参数集中在这里。
   */
  orb: {
    baseRadius: 0.34,        // 光球基础半径（实际半径 = baseRadius × 每颗随机尺寸系数）
    sizeJitter: 0.5,         // 直径随机浮动：1.0 ~ (1.0 + sizeJitter) 倍
    tints: ['#ff8fc0', '#ffa8d4', '#ffc2e2', '#ff9ecb', '#ffb6dc', '#ff7fb4'], // 与树同色系的粉（循环）
    rimColor: '#eaf3ff',     // 相册高亮时才显现的极淡浅蓝白边缘光
    spinSpeed: 0.35,         // 自转速度（呼吸/微旋相位）
    floatAmp: 0.22,          // 上下漂浮幅度
    floatSpeed: 1.5,         // 上下漂浮速度
    breatheAmp: 0.08,        // 呼吸缩放幅度
    photoStrengthTree: 0.06, // 树上时照片可见度（几乎看不见，只是光核里一点暖色）
    photoStrengthAlbum: 0.95,// 相册模式下照片可见度
    dimOpacity: 0.5,         // 未选中光球透明度系数（越小越淡）
    selectedScale: 1.3,      // 当前光球相对放大倍率
    highlightGlow: 1.5,      // 当前光球高亮增益
  },

  /* ---------------- 相册交互 / 过渡 ---------------- */
  album: {
    swipeRatio: 0.17,       // 滑动翻页阈值：占画面宽度比例（15%-20%）
    swipeCooldown: 800,     // 翻页防抖冷却（毫秒）
    flipDuration: 2.4,      // 翻页动画速度（越大越快，2.4 ≈ 0.42 秒）
    frameGlow: 0.9,         // 相框发光边框亮度
    photoRatio: 0.75,       // 竖版相框宽/高比（0.75 = 3:4）
    photoHeight: 9.6,       // 竖版相框高度（世界单位）
    burstStrength: 0.85,    // 进入/退出时粒子炸开强度
    burstTime: 260,         // 炸开阶段时长（毫秒）
    confettiDrift: 0.28,    // 彩屑飘落速度
  },

  /* ---------------- 背景音乐 ----------------
   * src 填你的音频文件路径即可（例如 './audio/christmas.mp3'），留空则不播放。
   * 浏览器策略：音频必须由一次"真实点击"解锁——角落的 ♪ 开关就是那次点击。
   * 解锁后，各场景手势只改变音量/氛围，不做生硬的播放停止，过渡更顺滑。
   */
  audio: {
    src: './audio/merry-christmas.mp3', // ★ 背景音乐文件（Merry Christmas · 新时代小星星组合）★
    volume: 0.38,            // 基础音量（0~1，作为背景氛围不宜太大）
    loop: true,              // 循环播放
    fadeInSec: 1.8,          // 淡入秒数
    fadeOutSec: 1.2,         // 静音淡出秒数
    fadeSceneSec: 1.0,       // 场景切换音量过渡秒数
    scenes: {                // 各场景音量倍率（相对 volume）
      tree: 1.0,             // 圣诞树：正常
      planet: 0.78,          // 星球：稍柔
      scatter: 0.62,         // 散开：更轻
      text: 1.15,            // 祝福文字：微微上扬
      album: 0.7,            // 相册：退后，不抢画面
      albumFlip: 0.5,        // 翻页瞬间的轻微 duck
    },
    flipDuckSec: 0.35,       // 翻页 duck 后恢复时长
  },

  autoReturnMs: 7000, // 文字显示多少毫秒后自动汇聚回圣诞树
  reduceMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
};

/* ========================= 2. DOM 引用 ========================= */
const dom = {
  container: document.getElementById('scene-container'),
  number: document.getElementById('hand-number'),
  label: document.getElementById('gesture-label'),
  panel: document.getElementById('start-panel'),
  note: document.getElementById('start-note'),
  video: document.getElementById('hand-video'),
  preview: document.getElementById('hand-preview'),
  previewWrap: document.getElementById('preview-wrap'),
  detectHud: document.getElementById('detect-hud'),
  cameraSwitch: document.getElementById('camera-switch'),
  delegateSwitch: document.getElementById('delegate-switch'),
  photoPage: document.getElementById('photo-page'),
  albumTip: document.getElementById('album-tip'),
  musicBtn: document.getElementById('music-toggle'),
};

/* ========================= 2.5 背景音乐模块 =========================
 * 用一个 <audio> + Web Audio 的 GainNode 做平滑音量过渡：
 *  - 角落 ♪ 开关承担"真实点击"，解锁 AudioContext 并首次播放（满足浏览器策略）。
 *  - 各场景手势只调目标音量（setScene），不中断播放，所以切换非常顺滑。
 *  - 没有音频文件（src 为空）时全部调用安全空转，绝不报错。
 */
const music = {
  ctx: null,
  gain: null,
  el: null,
  scene: 'tree',
  playing: false,
  ready: false,

  /** 首次用户点击时初始化：创建 AudioContext + 媒体源 + 增益节点 */
  init() {
    if (this.ready || !CONFIG.audio.src) return this.ready;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) throw new Error('浏览器不支持 Web Audio');
      this.ctx = new AudioCtx();
      this.gain = this.ctx.createGain();
      this.gain.gain.value = 0;
      this.gain.connect(this.ctx.destination);

      this.el = new Audio();
      this.el.src = CONFIG.audio.src;
      this.el.loop = CONFIG.audio.loop;
      this.el.preload = 'auto';
      // 网络音频需开启 CORS；本地/同源无影响
      if (/^https?:/i.test(CONFIG.audio.src)) this.el.crossOrigin = 'anonymous';

      const source = this.ctx.createMediaElementSource(this.el);
      source.connect(this.gain);

      this.el.addEventListener('ended', () => {
        if (CONFIG.audio.loop && this.playing) this.el.play().catch(() => {});
      });
      this.el.addEventListener('error', () => {
        console.warn('[音乐] 音频加载失败，请检查 CONFIG.audio.src：', CONFIG.audio.src);
        this.playing = false;
        this.updateButton();
      });

      this.ready = true;
      return true;
    } catch (err) {
      console.warn('[音乐] 初始化失败：', err);
      return false;
    }
  },

  /** 把音量平滑过渡到 target（0~1） */
  fadeTo(target, seconds) {
    if (!this.ready) return;
    const g = this.gain.gain;
    const now = this.ctx.currentTime;
    const v = Math.max(0, Math.min(1, target));
    if (typeof g.cancelAndHoldAtTime === 'function') {
      g.cancelAndHoldAtTime(now); // 从当前实际音量接续，避免跳变
    } else {
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
    }
    g.linearRampToValueAtTime(v, now + Math.max(0.01, seconds));
  },

  /** 当前场景应有的音量（未播放时为 0） */
  targetVolume() {
    if (!this.playing) return 0;
    const mult = CONFIG.audio.scenes[this.scene] ?? 1;
    return CONFIG.audio.volume * mult;
  },

  /** 场景切换：只改目标音量，播放不中断 */
  setScene(name) {
    if (!CONFIG.audio.scenes[name]) return;
    this.scene = name;
    if (this.playing) this.fadeTo(this.targetVolume(), CONFIG.audio.fadeSceneSec);
  },

  /** 翻页瞬间的轻微 duck，随后恢复当前场景音量 */
  duck() {
    if (!this.playing) return;
    const base = this.targetVolume();
    const duckV = CONFIG.audio.volume * (CONFIG.audio.scenes.albumFlip ?? 0.5);
    this.fadeTo(duckV, 0.12);
    clearTimeout(this._duckTimer);
    this._duckTimer = setTimeout(() => {
      this.fadeTo(this.targetVolume(), CONFIG.audio.flipDuckSec); // 用当前场景音量恢复
    }, 180);
  },

  /** 由「开启摄像头」等真实用户手势调用：幂等地解锁并播放音乐。
   *  浏览器只允许在用户手势栈里 play()，所以必须在点击处理函数内直接调用，
   *  不能在 await 摄像头/模型加载之后再调——那时手势栈已失效。 */
  ensurePlaying() {
    if (!CONFIG.audio.src || this.playing) return;
    if (!this.init()) return;
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    const p = this.el.play();
    if (p && typeof p.catch === 'function') p.catch((err) => console.warn('[音乐] 自动播放被拦截：', err));
    this.playing = true;
    this.gain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.gain.gain.setValueAtTime(0, this.ctx.currentTime);
    this.fadeTo(this.targetVolume(), CONFIG.audio.fadeInSec);
    this.updateButton();
  },

  /** 点击 ♪ 开关：解锁 + 播放 / 静音（这是浏览器认可的真实用户手势） */
  async toggle() {
    if (!CONFIG.audio.src) {
      setLabel('♪ 音乐待配置：在 CONFIG.audio.src 填入音频路径');
      return;
    }
    if (!this.init()) return;
    try {
      if (this.ctx.state === 'suspended') await this.ctx.resume();
      if (!this.playing) {
        await this.el.play();          // 必须在用户手势的调用栈内
        this.playing = true;
        // 从 0 淡入，避免"啪"一下；直接 setValueAtTime 保证起点确定
        this.gain.gain.cancelScheduledValues(this.ctx.currentTime);
        this.gain.gain.setValueAtTime(0, this.ctx.currentTime);
        this.fadeTo(this.targetVolume(), CONFIG.audio.fadeInSec);
      } else {
        this.fadeTo(0, CONFIG.audio.fadeOutSec);
        this.playing = false;
        clearTimeout(this._stopTimer);
        this._stopTimer = setTimeout(() => { if (!this.playing) this.el.pause(); },
          CONFIG.audio.fadeOutSec * 1000 + 80);
      }
      this.updateButton();
    } catch (err) {
      console.warn('[音乐] 播放失败：', err);
    }
  },

  updateButton() {
    if (!dom.musicBtn) return;
    const on = this.playing;
    dom.musicBtn.classList.toggle('is-on', on);
    dom.musicBtn.setAttribute('aria-label', on ? '关闭背景音乐' : '开启背景音乐');
    dom.musicBtn.title = on ? '关闭背景音乐 (M)' : '开启背景音乐 (M)';
  },
};

/* ========================= 3. 场景基础 ========================= */
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x140720, 0.011);

const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(0, 1.2, 23);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0); // 透明，让 CSS 的星空渐变背景透出
dom.container.appendChild(renderer.domElement);

/* ========================= 4. 工具：径向渐变纹理 ========================= */
function makeRadialTexture(stops) {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  stops.forEach(([offset, color]) => grad.addColorStop(offset, color));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/* ========================= 5. 星空背景粒子 ========================= */
function createStars() {
  const count = CONFIG.starCount;
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const speeds = new Float32Array(count);
  const phases = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 150;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
    positions[i * 3 + 2] = -20 - Math.random() * 90; // 全部放在树后方
    sizes[i] = 1 + Math.random() * 2.4;
    speeds[i] = 0.3 + Math.random() * 0.9;
    phases[i] = Math.random() * Math.PI * 2;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('aSpeed', new THREE.BufferAttribute(speeds, 1));
  geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: renderer.getPixelRatio() },
    },
    vertexShader: /* glsl */ `
      uniform float uTime;
      uniform float uPixelRatio;
      attribute float aSize;
      attribute float aSpeed;
      attribute float aPhase;
      varying float vTwinkle;

      void main() {
        vec3 p = position;
        // 缓慢漂浮：低频正弦位移
        p.y += sin(uTime * aSpeed * 0.35 + aPhase) * 1.6;
        p.x += cos(uTime * aSpeed * 0.22 + aPhase) * 0.8;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = aSize * uPixelRatio * (180.0 / -mv.z);
        vTwinkle = 0.55 + 0.45 * sin(uTime * (1.0 + aSpeed) + aPhase);
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vTwinkle;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        if (d > 0.5) discard;
        float a = 1.0 - smoothstep(0.0, 0.5, d);
        gl_FragColor = vec4(vec3(1.0, 0.90, 0.97), a * vTwinkle * 0.85);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  scene.add(points);
  return mat;
}
const starMat = createStars();

/* ========================= 5.5 礼物相册：纹理 / 光球大粒子 / 彩屑 / 相框 =========================
 * 每个光球与 CONFIG.photos 一一对应：不是独立玻璃球，而是"粒子树里恰好较大、较亮的一颗"。
 * 用与主粒子一致的加法混合径向光 + 极淡浅蓝白边缘光 + 半封的圆形照片；
 * 平时挂在树上不突兀，进入相册后漂浮在彩屑场中，当前照片光球高亮脉冲。
 */
const PHOTOS = CONFIG.photos;
const PHOTO_COUNT = PHOTOS.length;
const ORB_TINTS = CONFIG.orb.tints;        // 光球粉白微染色（循环）
const CONFETTI_COLORS = CONFIG.confettiColors; // 彩屑/占位图撞色（循环）

/** 生成竖版渐变占位纹理（当照片 src 为空或加载失败时使用，自带星云漩涡 + 标题） */
function makePhotoTexture(title, index) {
  const W = 480, H = 640; // 竖版 3:4
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const c1 = CONFETTI_COLORS[index % CONFETTI_COLORS.length];
  const c2 = CONFETTI_COLORS[(index + 2) % CONFETTI_COLORS.length];
  // 深色底渐变
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#170a24');
  grad.addColorStop(0.55, '#2a1038');
  grad.addColorStop(1, '#0d0718');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
  // 多层撞色漩涡光斑
  for (let k = 0; k < 6; k++) {
    const a = k * 1.05 + index * 0.7;
    const cx = W * 0.5 + Math.cos(a) * W * 0.3;
    const cy = H * 0.44 + Math.sin(a) * H * 0.26;
    const rad = W * (0.34 + (k % 3) * 0.08);
    const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, rad);
    const col = k % 2 ? c1 : c2;
    g.addColorStop(0, col + 'bb');
    g.addColorStop(0.45, col + '44');
    g.addColorStop(1, 'transparent');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }
  // 星点
  ctx.globalAlpha = 0.6;
  for (let i = 0; i < 70; i++) {
    const x = (i * 137.5) % W, y = (i * 89.3) % H;
    ctx.fillStyle = i % 2 ? c2 : '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, (i % 3) * 0.9 + 0.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  // 中心玻璃光球图案（呼应树上的光球）
  const cx = W / 2, cy = H / 2 - 26, R = 92;
  const orb = ctx.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.05, cx, cy, R);
  orb.addColorStop(0, 'rgba(255,255,255,0.95)');
  orb.addColorStop(0.25, c1 + 'cc');
  orb.addColorStop(0.7, c2 + '99');
  orb.addColorStop(1, 'rgba(255,255,255,0.05)');
  ctx.fillStyle = orb;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  // 标题
  ctx.textAlign = 'center';
  ctx.fillStyle = '#fff4fa';
  ctx.font = 'bold 30px "Microsoft YaHei", sans-serif';
  ctx.shadowColor = c1; ctx.shadowBlur = 18;
  ctx.fillText(title || `照片 ${index + 1}`, W / 2, H - 46);
  ctx.shadowBlur = 0;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * "大粒子"光球材质：与主粒子系统同样用加法混合的柔和径向光，只是更大、更亮，
 * 中心封着一张若隐若现的圆形照片。极淡浅蓝白边缘光 + 一点高光模拟肥皂泡，
 * 但没有实心玻璃球的硬镜面与撞色漩涡，因此挂在树上时读作"较大一点的粒子"而非突兀装饰。
 */
function makeOrbMaterial(photoTexture, tint, rimColor) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPhoto: { value: photoTexture },
      uTint: { value: new THREE.Color(tint) },
      uRim: { value: new THREE.Color(rimColor) },
      uHighlight: { value: 0 },  // 0=普通 1=当前照片高亮
      uDim: { value: 1 },        // 未选中透明度系数
      uAlbum: { value: 0 },      // 0=树上(纯发光粒子) 1=相册(显出泡泡边缘光)
      uPhotoStrength: { value: CONFIG.orb.photoStrengthTree }, // 照片可见度（树上弱/相册强）
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uPhoto;
      uniform vec3 uTint;
      uniform vec3 uRim;
      uniform float uHighlight;
      uniform float uDim;
      uniform float uAlbum;
      uniform float uPhotoStrength;
      varying vec2 vUv;
      void main() {
        vec2 p = vUv - 0.5;
        float r = length(p) * 2.0;
        if (r > 1.0) discard;

        // 柔和宽广的径向光核：像一颗"被放大的粒子"，而不是实心玻璃球
        float core = 1.0 - smoothstep(0.0, 1.0, r);
        float halo = pow(core, 1.4);
        // 中心一点更亮的核
        float nucleus = pow(1.0 - smoothstep(0.0, 0.42, r), 2.0);
        // 边缘柔化：让光点没有硬边，自然融进粒子里
        float feather = 1.0 - smoothstep(0.72, 1.0, r);
        // 极淡浅蓝白边缘光 + 高光：只在相册里显现，树上完全不出现，避免"玻璃珠"感
        float rim = smoothstep(0.62, 0.88, r) * (1.0 - smoothstep(0.88, 1.0, r));
        vec2 hp = p - vec2(-0.20, 0.24);
        float spec = exp(-dot(hp, hp) * 140.0) * 0.35;

        // 圆形照片：加法式淡淡透出，绝不把光核压暗（否则中心发黑像玻璃珠）
        vec2 uv = p / 0.62 + 0.5;
        uv.y = uv.y * 0.76 + 0.12;
        vec4 tex = texture2D(uPhoto, uv);
        float photoMask = 1.0 - smoothstep(0.50, 0.62, r);
        float pw = photoMask * uPhotoStrength;

        vec3 glow = uTint * (halo + nucleus * 0.55) * feather;
        glow += uRim * rim * uAlbum * (0.20 + uHighlight * 0.55);
        glow += vec3(1.0) * spec * uAlbum;
        glow += tex.rgb * pw * 1.1;
        float a = (halo * 0.82 + pw * 0.25) * feather * uDim * (0.85 + uHighlight * 0.4);
        gl_FragColor = vec4(glow, clamp(a, 0.0, 1.0));
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: false,            // 始终叠在粒子之上，像一颗更亮的大粒子
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
}

/** 供纹理异步加载完成后同步替换光球内的照片 */
const orbMaterials = [];

/** 为每张照片准备纹理：优先加载 src，失败/空则用兜底纹理，绝不抛错 */
const photoTextures = PHOTOS.map((p, i) => {
  const fallback = makePhotoTexture(p.title, i);
  if (!p.src) return fallback;
  const loader = new THREE.TextureLoader();
  loader.load(
    p.src,
    (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      photoTextures[i]?.dispose?.();
      photoTextures[i] = tex;
      if (photoPlane && albumState.photoIndex === i) photoPlane.material.map = tex;
      if (orbMaterials[i]) orbMaterials[i].uniforms.uPhoto.value = tex; // 同步光球内照片
    },
    undefined,
    () => console.warn(`[相册] 照片加载失败，使用占位纹理：${p.src}`)
  );
  return fallback;
});

/* ---------- 光球大粒子（与照片一一对应）---------- */
const giftOrbs = [];
const orbGroup = new THREE.Group();
scene.add(orbGroup);
const orbGeo = new THREE.PlaneGeometry(1, 1); // 相机朝向的方形面片，shader 里画成圆形光点

for (let i = 0; i < PHOTO_COUNT; i++) {
  const tint = ORB_TINTS[i % ORB_TINTS.length];
  const group = new THREE.Group();

  // 一颗"更大更亮的粒子"：加法混合径向光 + 极淡边缘光 + 半封照片
  const mat = makeOrbMaterial(photoTextures[i], tint, CONFIG.orb.rimColor);
  const mesh = new THREE.Mesh(orbGeo, mat);
  group.add(mesh);
  orbMaterials[i] = mat;

  // 每颗直径略有差异：1.0 ~ (1 + sizeJitter) 倍
  const sizeK = 1 + Math.random() * CONFIG.orb.sizeJitter;
  const radius = CONFIG.orb.baseRadius * sizeK;

  orbGroup.add(group);

  // 树上的挂点：沿高度错落分布，挂在树冠表面外侧
  const t = (i + 0.5) / PHOTO_COUNT;                 // 0..1 自下而上
  const h = 0.12 + t * 0.78;                         // 避开顶点和底边
  const y = h * CONFIG.tree.height - CONFIG.tree.height * 0.5;
  const rr = CONFIG.tree.radius * (1 - h) + 0.35;    // 略在树冠外
  const angle = i * 2.39996;                          // 黄金角，分布均匀不重叠
  const treePos = new THREE.Vector3(Math.cos(angle) * rr, y, Math.sin(angle) * rr);

  giftOrbs.push({
    group, mesh, mat, treePos, radius, tint,
    phase: Math.random() * Math.PI * 2,
    spin: Math.random() * Math.PI * 2,
    current: treePos.clone(),
  });
}

/* ---------- 相册全屏彩屑粒子场（碎金箔感，随相册形态淡入淡出）---------- */
function createConfetti() {
  const count = CONFIG.confettiCount;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const speeds = new Float32Array(count);
  const palette = CONFETTI_COLORS.map((c) => new THREE.Color(c));
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 46;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 32;
    positions[i * 3 + 2] = -10 + Math.random() * 18;
    const c = palette[i % palette.length];
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    sizes[i] = 1.6 + Math.random() * 2.8;
    phases[i] = Math.random() * Math.PI * 2;
    speeds[i] = 0.5 + Math.random() * 1.2;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  geo.setAttribute('aSpeed', new THREE.BufferAttribute(speeds, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uReveal: { value: 0 }, // 相册形态权重：0 隐藏，1 全屏
      uPixelRatio: { value: renderer.getPixelRatio() },
      uDrift: { value: CONFIG.album.confettiDrift },
    },
    vertexShader: /* glsl */ `
      uniform float uTime, uReveal, uPixelRatio, uDrift;
      attribute vec3 aColor;
      attribute float aSize, aPhase, aSpeed;
      varying vec3 vColor;
      varying float vPhase;
      void main() {
        vec3 p = position;
        // 缓慢循环飘落 + 水平摆动
        p.y = mod(p.y - uTime * aSpeed * uDrift * 3.0 + 16.0, 32.0) - 16.0;
        p.x += sin(uTime * 0.6 + aPhase) * 0.9;
        vColor = aColor; vPhase = aPhase;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = aSize * uPixelRatio * (110.0 / -mv.z) * (0.35 + uReveal * 0.65);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime, uReveal;
      varying vec3 vColor;
      varying float vPhase;
      void main() {
        if (uReveal < 0.001) discard;
        vec2 uv = gl_PointCoord - 0.5;
        // 箔片自转
        float a = vPhase + uTime * 1.4;
        float cs = cos(a), sn = sin(a);
        uv = mat2(cs, -sn, sn, cs) * uv;
        // 菱形箔片
        float shape = max(abs(uv.x) * 1.5, abs(uv.y) * 2.2);
        if (shape > 0.5) discard;
        float hi = 1.0 - smoothstep(0.0, 0.5, shape);
        float tw = 0.65 + 0.35 * sin(uTime * 3.2 + vPhase * 3.0);
        gl_FragColor = vec4(vColor * (0.75 + hi * 0.9) * tw, (1.0 - shape * 1.6) * uReveal * 0.85);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  scene.add(points);
  return mat;
}
const confettiMat = createConfetti();

/* ---------- 相册模式状态（纹理回调里会用到，提前声明） ---------- */
const albumState = {
  active: false,       // 是否处于相册模式
  photoIndex: 0,       // 当前照片索引
  morph: 0,            // 0=树上装饰，1=相册排布（平滑过渡）
  target: 0,
  flipPhase: 0,        // 翻页动画进度 0→1
  flipDir: 0,          // 1=下一张，-1=上一张
  flipping: false,
  burstTimer: null,    // 收拢/炸开编排定时器
};
let photoPlane = null; // 纹理加载回调可能早于平面创建，先占位

/* ---------- 竖版悬浮相框：浅色发光边框 + 照片平面（支持模糊翻页）---------- */
const PHOTO_PLANE_H = CONFIG.album.photoHeight;              // 竖版高度
const PHOTO_PLANE_W = PHOTO_PLANE_H * CONFIG.album.photoRatio; // 竖版宽度（3:4）

// 浅色发光边框（置于照片后方一圈，营造柔和光晕）
const frameGlow = new THREE.Mesh(
  new THREE.PlaneGeometry(PHOTO_PLANE_W + 1.1, PHOTO_PLANE_H + 1.1),
  new THREE.MeshBasicMaterial({
    color: 0xffe9f8, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  })
);
frameGlow.position.set(0, 0.5, 2.86);
frameGlow.visible = false;
scene.add(frameGlow);

// 模糊 uniform：注入到 MeshBasicMaterial，翻页时调节 uBlur 实现由虚到实
const photoBlur = { value: 0 };
photoPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(PHOTO_PLANE_W, PHOTO_PLANE_H),
  new THREE.MeshBasicMaterial({
    map: photoTextures[0],
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
);
photoPlane.material.onBeforeCompile = (shader) => {
  shader.uniforms.uBlur = photoBlur;
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>
      uniform float uBlur;
      vec4 blurSample(sampler2D tex, vec2 uv, float r) {
        vec4 c = texture2D(tex, uv) * 0.28;
        c += texture2D(tex, uv + vec2( r, 0.0)) * 0.12;
        c += texture2D(tex, uv + vec2(-r, 0.0)) * 0.12;
        c += texture2D(tex, uv + vec2(0.0,  r)) * 0.12;
        c += texture2D(tex, uv + vec2(0.0, -r)) * 0.12;
        c += texture2D(tex, uv + vec2( r,  r)) * 0.06;
        c += texture2D(tex, uv + vec2(-r,  r)) * 0.06;
        c += texture2D(tex, uv + vec2( r, -r)) * 0.06;
        c += texture2D(tex, uv + vec2(-r, -r)) * 0.08;
        return c;
      }`)
    .replace('#include <map_fragment>', `
      #ifdef USE_MAP
        vec4 sampledDiffuseColor = blurSample(map, vMapUv, uBlur * 0.02);
        diffuseColor *= sampledDiffuseColor;
      #endif
    `);
};
photoPlane.material.customProgramCacheKey = () => 'photo-blur-v1';
photoPlane.position.set(0, 0.5, 3);
photoPlane.visible = false;
scene.add(photoPlane);

/* ---------- 相册粒子目标点 aAlbum：竖版相框边框 + 后方旋转环带 ---------- */
function buildAlbumAttributes(count) {
  const album = new Float32Array(count * 3);
  const FW = PHOTO_PLANE_W * 0.5 + 0.6;  // 相框半宽
  const FH = PHOTO_PLANE_H * 0.5 + 0.6;  // 相框半高
  const perim = 2 * (FW * 2 + FH * 2);
  for (let i = 0; i < count; i++) {
    let x, y, z;
    if (i % 10 < 4) {
      // 40% 粒子沿竖版照片四周排成粒子相框
      const d = (i / count) * perim * 2.5 % perim;
      if (d < FW * 2) { x = -FW + d; y = FH; }
      else if (d < FW * 2 + FH * 2) { x = FW; y = FH - (d - FW * 2); }
      else if (d < FW * 4 + FH * 2) { x = FW - (d - FW * 2 - FH * 2); y = -FH; }
      else { x = -FW; y = -FH + (d - FW * 4 - FH * 2); }
      x += (Math.random() - 0.5) * 0.18;
      y += (Math.random() - 0.5) * 0.18;
      z = 3.2;
    } else {
      // 60% 粒子汇聚成照片后方的旋转环带，营造"照片悬浮"的科技感
      const a = Math.random() * Math.PI * 2;
      const r = 9 + Math.random() * 3.5;
      x = Math.cos(a) * r;
      y = Math.sin(a) * r * 0.62 + 0.5;
      z = -2 - Math.random() * 4;
    }
    album[i * 3] = x;
    album[i * 3 + 1] = y;
    album[i * 3 + 2] = z;
  }
  return album;
}

/* ========================= 6. 文字采样 → 粒子目标点 =========================
 * 用 Canvas 2D 把中文文字画成位图，再逐像素采样出亮点的坐标，
 * 转成 3D 世界坐标，作为粒子 morph 的目标位置。
 */
function sampleTextPoints(text, count) {
  const W = 1200, H = 420;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  // 用 alphabetic 基线并把基线放在画布偏下位置，保证中文笔画完整落在画布内。
  // 之前用 middle 基线会把中文字形切掉一半（采样到的就是半截字）。
  ctx.textBaseline = 'alphabetic';

  // 自动缩放字号，长文字也不会溢出
  let fontSize = 170;
  const font = (s) => `bold ${s}px "Microsoft YaHei", "PingFang SC", sans-serif`;
  ctx.font = font(fontSize);
  const measured = ctx.measureText(text).width;
  if (measured > W - 80) {
    fontSize = Math.max(48, Math.floor(fontSize * (W - 80) / measured));
    ctx.font = font(fontSize);
  }
  // 基线放在画布 78% 高度处：中文主体在基线上方，顶部和底部都留足空间
  ctx.fillText(text, W / 2, H * 0.78);

  const data = ctx.getImageData(0, 0, W, H).data;
  const candidates = [];
  for (let y = 0; y < H; y += 3) {
    for (let x = 0; x < W; x += 3) {
      if (data[(y * W + x) * 4 + 3] > 100) candidates.push([x, y]);
    }
  }
  // 极端兜底（如字体缺失导致无像素）
  if (candidates.length === 0) {
    for (let i = 0; i < count; i++) candidates.push([100 + (i / count) * 1000, H / 2]);
  }

  // 关键：按实际采样到的像素边界归一化，而不是按画布中心。
  // 中文字形的墨迹范围并不以 textBaseline 对称，用画布中心映射会丢掉一半笔画。
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of candidates) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const targetW = CONFIG.text.size;          // 世界宽度（在 CONFIG.text.size 调整字号大小）
  const targetH = targetW * spanY / spanX;   // 按字形真实宽高比推算，不变形

  // 随机打乱候选点后取前 count 个：保证字形上/中/下各区域都被均匀覆盖。
  // 绝不能用 i % candidates.length：candidates 按 y 从上到下扫描排列，当粒子数
  // 少于候选像素数时只会取到文字上半部分，字形被砍成横条（之前「半截字」的真正根因）。
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp;
  }

  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const pt = candidates[i % candidates.length];
    // 把 [minX, maxX] / [minY, maxY] 完整映射到 [-targetW/2, +targetW/2] / [-targetH/2, +targetH/2]
    out[i * 3] = ((pt[0] - minX) / spanX - 0.5) * targetW + (Math.random() - 0.5) * 0.06;
    out[i * 3 + 1] = (0.5 - (pt[1] - minY) / spanY) * targetH + (Math.random() - 0.5) * 0.06;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.5;
  }
  return { positions: out, heightRatio: spanY / spanX }; // heightRatio 供 shader 精确还原字形比例
}

/* ========================= 7. 圣诞树粒子系统（核心） ========================= */
function buildTreeAttributes(count) {
  const tree = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const randoms = new Float32Array(count);
  const { height: H, radius: R, arms: ARMS, turns: TURNS } = CONFIG.tree;

  // 底部玫粉 → 中部腮红粉 → 顶部浅粉白（配色在 CONFIG.tree.colors 调整）
  const c1 = new THREE.Color(CONFIG.tree.colors[0]);
  const c2 = new THREE.Color(CONFIG.tree.colors[1]);
  const c3 = new THREE.Color(CONFIG.tree.colors[2]);
  const tmp = new THREE.Color();

  for (let i = 0; i < count; i++) {
    const h = Math.pow(Math.random(), 0.72);      // 高度比例，底部略密
    const y = h * H - H * 0.5;
    const rr = R * (1 - h);                       // 圆锥：越往上半径越小
    let angle, r;

    if (Math.random() < 0.78) {
      // 78% 粒子落在螺旋臂上，形成清晰的螺旋纹理
      const arm = i % ARMS;
      angle = h * Math.PI * 2 * TURNS + (arm / ARMS) * Math.PI * 2 + (Math.random() - 0.5) * 0.14;
      r = rr * (0.9 + Math.random() * 0.18);
    } else {
      // 其余粒子随机填充圆锥面，让树更饱满
      angle = Math.random() * Math.PI * 2;
      r = rr * Math.sqrt(Math.random()) * 0.98;
    }

    tree[i * 3] = r * Math.cos(angle);
    tree[i * 3 + 1] = y;
    tree[i * 3 + 2] = r * Math.sin(angle);
    randoms[i] = Math.random();

    if (h < 0.55) tmp.copy(c1).lerp(c2, h / 0.55);
    else tmp.copy(c2).lerp(c3, (h - 0.55) / 0.45);
    colors[i * 3] = tmp.r;
    colors[i * 3 + 1] = tmp.g;
    colors[i * 3 + 2] = tmp.b;
  }
  return { tree, colors, randoms };
}

/* ---------- 星球形态数据：球面本体 + 倾斜星环 ----------
 * 复用同一批粒子，为它们生成另一套目标位置 aPlanet。
 * 每个粒子带一个 aPart 标记：0=星球表面，1=星环内圈，2=星环外圈。
 * 球面用 Fibonacci 均匀采样，并用低频噪声做陆地/云层明暗层次。
 */
function buildPlanetAttributes(count) {
  const planet = new Float32Array(count * 3);
  const part = new Float32Array(count);      // 0=球面 1=内环 2=外环
  const shade = new Float32Array(count);     // 明暗层次 0..1
  const ringInfo = new Float32Array(count * 2); // 环粒子：[极角, 半径]，球面粒子为 0
  const R = 3.6;                              // 星球半径
  const golden = Math.PI * (3 - Math.sqrt(5)); // 黄金角

  for (let i = 0; i < count; i++) {
    if (i % 10 < 7) {
      // ---- 70%：星球表面（Fibonacci 球面均匀分布）----
      part[i] = 0;
      const t = (i + 0.5) / count;
      const y = 1 - 2 * t;                    // -1..1
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * i;
      const px = Math.cos(theta) * r;
      const pz = Math.sin(theta) * r;
      // 低频伪噪声：模拟大陆/云层的明暗块
      const n = Math.sin(px * 4.1 + y * 3.3) * Math.cos(pz * 3.7 - y * 2.2)
              + 0.5 * Math.sin(px * 8.3 + pz * 6.1 + y * 5.0);
      shade[i] = Math.max(0, Math.min(1, 0.5 + n * 0.35));
      // 球面粒子略微分层，增加体积感
      const rad = R * (0.97 + shade[i] * 0.06);
      planet[i * 3] = px * rad;
      planet[i * 3 + 1] = y * rad;
      planet[i * 3 + 2] = pz * rad;
    } else {
      // ---- 30%：星环（两条倾斜圆环）----
      const outer = (i % 20 < 17) ? 1 : 0;   // 内环密、外环疏
      part[i] = outer ? 2 : 1;
      const ringR = outer ? R * 1.75 : R * 1.42;
      const spread = outer ? 0.34 : 0.22;    // 环宽度
      const a = Math.random() * Math.PI * 2;
      const rr = ringR + (Math.random() - 0.5) * spread * 2;
      ringInfo[i * 2] = a;      // 存极角，shader 里 +自转 重建位置，倾角恒定
      ringInfo[i * 2 + 1] = rr; // 存半径
      // 初始位置（shader 会用 ringInfo 重算，这里给个初值即可）
      let x = Math.cos(a) * rr;
      let z = Math.sin(a) * rr;
      let y = 0;
      const ix = 25 * Math.PI / 180;
      const iy = y * Math.cos(ix) - z * Math.sin(ix);
      const iz = y * Math.sin(ix) + z * Math.cos(ix);
      y = iy; z = iz;
      const iz2ang = 12 * Math.PI / 180;
      const x2 = x * Math.cos(iz2ang) - y * Math.sin(iz2ang);
      const y2 = x * Math.sin(iz2ang) + y * Math.cos(iz2ang);
      planet[i * 3] = x2;
      planet[i * 3 + 1] = y2;
      planet[i * 3 + 2] = z;
      shade[i] = 0.6 + Math.random() * 0.4;
    }
  }
  return { planet, part, shade, ringInfo };
}

const COUNT = CONFIG.particleCount;
const { tree: treePos, colors: treeColors, randoms: treeRandoms } = buildTreeAttributes(COUNT);
const textA = sampleTextPoints(CONFIG.text.first, COUNT);
const textB = sampleTextPoints(CONFIG.text.second, COUNT);
const { planet: planetPos, part: planetPart, shade: planetShade, ringInfo } = buildPlanetAttributes(COUNT);
const albumPos = buildAlbumAttributes(COUNT);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(treePos, 3)); // 仅作包围盒占位
geometry.setAttribute('aTree', new THREE.BufferAttribute(treePos, 3));
geometry.setAttribute('aTextA', new THREE.BufferAttribute(textA.positions, 3));
geometry.setAttribute('aTextB', new THREE.BufferAttribute(textB.positions, 3));
geometry.setAttribute('aPlanet', new THREE.BufferAttribute(planetPos, 3));
geometry.setAttribute('aAlbum', new THREE.BufferAttribute(albumPos, 3));
geometry.setAttribute('aPart', new THREE.BufferAttribute(planetPart, 1));
geometry.setAttribute('aShade', new THREE.BufferAttribute(planetShade, 1));
geometry.setAttribute('aRingInfo', new THREE.BufferAttribute(ringInfo, 2));
geometry.setAttribute('aColor', new THREE.BufferAttribute(treeColors, 3));
geometry.setAttribute('aRand', new THREE.BufferAttribute(treeRandoms, 1));

const particleMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uMorph: { value: 0 },       // 0 = 圣诞树, 1 = 文字
    uScatter: { value: 0 },     // 散开程度
    uJitter: { value: 0 },      // 抖动强度
    uTextIndex: { value: 0 },   // 0 = 文字A, 1 = 文字B
    uTextScale: { value: 1 },   // 文字自适应缩放（窄屏不裁切，x/y 同系数不改字形）
    uPlanetMorph: { value: 0 }, // 0 = 树/文字, 1 = 星球
    uPlanetSpin: { value: 0 },  // 星球自转角
    uAlbumMorph: { value: 0 },  // 0 = 树/星球, 1 = 相册（粒子汇聚成相框+环带）
    uTreeRotation: { value: 0 },
    uSize: { value: CONFIG.particleSize },   // 粒子基础尺寸（个体大小总开关，越大光点越饱满）
    uGlowStrength: { value: CONFIG.glowStrength }, // 粒子发光强度增益
    uPixelRatio: { value: renderer.getPixelRatio() },
  },
  vertexShader: /* glsl */ `
    uniform float uTime;
    uniform float uMorph;
    uniform float uScatter;
    uniform float uJitter;
    uniform float uTextIndex;
    uniform float uTextScale;
    uniform float uPlanetMorph;
    uniform float uPlanetSpin;
    uniform float uAlbumMorph;
    uniform float uTreeRotation;
    uniform float uSize;
    uniform float uPixelRatio;

    attribute vec3 aTree;
    attribute vec3 aTextA;
    attribute vec3 aTextB;
    attribute vec3 aColor;
    attribute float aRand;
    attribute vec3 aPlanet;
    attribute vec3 aAlbum;   // 相册形态目标位置（相框+环带）
    attribute float aPart;   // 0=球面 1=内环 2=外环
    attribute float aShade;  // 明暗层次
    attribute vec2 aRingInfo; // 环粒子：[极角, 半径]，用于倾角恒定的自转重建

    varying vec3 vColor;
    varying float vAlpha;
    varying float vShade;
    varying float vTextMix; // 文字形态权重：用于片元阶段让文字粒子更实心清晰

    void main() {
      // 1) 树形态：绕 Y 轴缓慢自转（文字形态不旋转，保持正对镜头）
      float s = sin(uTreeRotation), c = cos(uTreeRotation);
      vec3 tree = aTree;
      tree.x = aTree.x * c - aTree.z * s;
      tree.z = aTree.x * s + aTree.z * c;

      // 散开方向与抖动（在各自形态上叠加）
      vec3 dir = vec3(
        fract(aRand * 91.7) - 0.5,
        fract(aRand * 47.3) - 0.5,
        fract(aRand * 73.1) - 0.5
      );
      dir = normalize(dir + vec3(0.001));
      vec3 shake = vec3(
        sin(uTime * 14.0 + aRand * 121.0),
        cos(uTime * 17.0 + aRand * 83.0),
        sin(uTime * 11.0 + aRand * 57.0)
      );

      // ===== 树的裁剪空间位置（正常透视投影）=====
      vec3 treePos = tree + dir * uScatter * (4.0 + aRand * 10.0) + shake * uJitter * 0.5;
      vec4 treeClip = projectionMatrix * modelViewMatrix * vec4(treePos, 1.0);

      // ===== 文字的裁剪空间位置 =====
      // 用与树相同的透视投影：x/y 共用同一个投影矩阵，字形比例在任何窗口宽高比下都恒定正确。
      // （之前用屏幕空间 NDC 手算 aspect 补偿，窄屏下补偿失效导致文字被压成横条。）
      vec3 target = mix(aTextA, aTextB, uTextIndex);
      // target.x ∈ [-7,7]，target.y 已按字形真实墨迹比例缩放（targetH = 14*spanY/spanX）
      target.xy *= uTextScale; // 窄屏按比例缩小，保证完整落在画面内（x/y 同系数不改字形）
      vec3 textPos = target + dir * uScatter * 3.0 + shake * uJitter * 0.5;
      vec4 textClip = projectionMatrix * modelViewMatrix * vec4(textPos, 1.0);

      // ===== 星球形态：球面自转 + 星环独立旋转（透视投影）=====
      vec3 pl = aPlanet;
      if (aPart < 0.5) {
        // 球面：绕 Y 轴自转
        float ps = sin(uPlanetSpin), pc = cos(uPlanetSpin);
        pl.x = aPlanet.x * pc - aPlanet.z * ps;
        pl.z = aPlanet.x * ps + aPlanet.z * pc;
      } else {
        // 星环：用 aRingInfo 重建，极角 + 自转，倾角恒定（不会像绕 Y 轴转那样左右晃动）
        float ang = aRingInfo.x + uPlanetSpin * 1.8; // 环绕自身轴旋转，比球面快
        float rr = aRingInfo.y;
        float rx = cos(ang) * rr;
        float rz = sin(ang) * rr;
        // 固定倾角：先绕 X 轴 25°，再绕 Z 轴 12°（与 buildPlanetAttributes 一致）
        float ix = 25.0 * 3.14159265 / 180.0;
        float iy = 0.0 * cos(ix) - rz * sin(ix);
        float iz = 0.0 * sin(ix) + rz * cos(ix);
        float izA = 12.0 * 3.14159265 / 180.0;
        float fx = rx * cos(izA) - iy * sin(izA);
        float fy = rx * sin(izA) + iy * cos(izA);
        pl = vec3(fx, fy, iz);
      }
      vec4 planetClip = projectionMatrix * modelViewMatrix * vec4(pl, 1.0);

      // ===== 相册形态：粒子汇聚成照片相框 + 后方环带（缓慢旋转）=====
      vec3 ap = aAlbum;
      // 仅环带部分（z<0）绕 Y 轴缓慢转动，相框保持固定
      if (ap.z < 0.0) {
        float aa = uTime * 0.12;
        float ax = ap.x * cos(aa) - ap.z * sin(aa);
        float az = ap.x * sin(aa) + ap.z * cos(aa);
        ap = vec3(ax, ap.y, az);
      }
      vec4 albumClip = projectionMatrix * modelViewMatrix * vec4(ap, 1.0);

      // 形态合成：树↔文字 → 星球 → 相册（每步平滑叠加）
      vec4 baseClip = mix(treeClip, textClip, uMorph);
      vec4 withPlanet = mix(baseClip, planetClip, uPlanetMorph);
      gl_Position = mix(withPlanet, albumClip, uAlbumMorph);

      // 点大小：树/文字/星球统一按透视缩放，距离一致时大小协调
      // 文字粒子额外放大 + 提高亮度权重，让笔画更实、更清晰（用户反馈"粒子变小"）
      float treeSize   = uSize * uPixelRatio * (60.0 / -treeClip.w)   * (0.70 + aRand * 0.9);
      float textSize   = uSize * uPixelRatio * (60.0 / -textClip.w)   * (1.15 + aRand * 1.1);
      float planetSize = uSize * uPixelRatio * (60.0 / -planetClip.w) * (aPart < 0.5 ? 1.0 : 0.6);
      float albumSize  = uSize * uPixelRatio * (60.0 / -albumClip.w)  * (ap.z >= 0.0 ? 0.62 : 0.42);
      float baseSize = mix(treeSize, textSize, uMorph);
      float sizeWithPlanet = mix(baseSize, planetSize, uPlanetMorph);
      gl_PointSize = mix(sizeWithPlanet, albumSize, uAlbumMorph);

      // 星球专用配色：暗部深玫粉、亮部浅粉白，形成陆地/云层/光斑的层次
      vec3 planetCol = mix(vec3(0.72, 0.16, 0.38), vec3(1.0, 0.84, 0.94), pow(aShade, 1.3));
      vColor = mix(aColor, planetCol, uPlanetMorph);
      // 文字形态提亮：vColor 增益让粒子文字更醒目，不随叠加混合被稀释
      vColor *= 1.0 + uMorph * 0.5;
      vAlpha = 0.78 + 0.22 * sin(uTime * 2.2 + aRand * 40.0);
      // 明暗：星球球面强化对比（暗部更暗、亮部更亮），其它形态为 1
      vShade = mix(1.0, 0.55 + aShade * 1.15, uPlanetMorph);
      // 星环：亮粉色调 + 转到星球背后时压暗，形成前后遮挡关系
      if (uPlanetMorph > 0.5 && aPart >= 0.5) {
        vColor = mix(vColor, vec3(1.0, 0.72, 0.9), 0.7);
        vShade *= smoothstep(-2.0, 1.5, pl.z) * 0.55 + 0.45;
      }
      // 文字形态标记（星球/相册形态时归零，避免影响其它渲染）
      vTextMix = uMorph * (1.0 - uPlanetMorph) * (1.0 - uAlbumMorph);
      // 相册形态：粒子采用紫/蓝/绿/金/浅粉撞色，与彩屑场、玻璃光球呼应
      if (uAlbumMorph > 0.001) {
        vec3 cA = vec3(0.69, 0.42, 1.00); // 紫
        vec3 cB = vec3(0.30, 0.72, 1.00); // 蓝
        vec3 cC = vec3(0.37, 0.88, 0.66); // 绿
        vec3 cD = vec3(1.00, 0.83, 0.36); // 金
        vec3 cE = vec3(1.00, 0.62, 0.78); // 浅粉
        float k = fract(aRand * 5.0);
        vec3 conf = cA;
        if (k >= 1.0) conf = cB;
        if (k >= 2.0) conf = cC;
        if (k >= 3.0) conf = cD;
        if (k >= 4.0) conf = cE;
        vColor = mix(vColor, conf, uAlbumMorph);
      }
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uGlowStrength;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vShade;
    varying float vTextMix;
    void main() {
      vec2 uv = gl_PointCoord - 0.5;
      float d = length(uv);
      if (d > 0.5) discard;
      // 树/星球：柔和光点（中心亮核）；文字：更实心的圆点，边缘衰减更缓，笔画更连贯清晰
      float softTree = 1.0 - smoothstep(0.06, 0.5, d);
      float softText = 1.0 - smoothstep(0.30, 0.5, d);
      float soft = mix(softTree, softText, vTextMix);
      float coreTree = pow(softTree, 3.0);
      float coreText = pow(softText, 1.4);
      float core = mix(coreTree, coreText, vTextMix);
      vec3 col = vColor * (0.55 + core * 1.5) * vShade * uGlowStrength;
      // 文字形态不透明度拉满，避免叠加混合后被背景稀释
      float alpha = mix(soft * vAlpha, min(1.0, soft * 1.35), vTextMix);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending, // 叠加混合产生发光光效
});

const treePoints = new THREE.Points(geometry, particleMaterial);
treePoints.frustumCulled = false; // 粒子会飞出原始包围盒，关闭视锥剔除
scene.add(treePoints);

/** 文字自适应：按相机可见宽度缩放，窄屏不裁切、宽屏不过小（x/y 同系数，不改字形比例） */
function updateTextScale() {
  const visibleHeight = 2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const visibleWidth = visibleHeight * camera.aspect;
  // 文字基础宽度由 CONFIG.text.size 决定，最多占可见宽度的 CONFIG.text.widthRatio
  const scale = (visibleWidth * CONFIG.text.widthRatio) / CONFIG.text.size;
  particleMaterial.uniforms.uTextScale.value = Math.max(0.28, Math.min(1.6, scale));
}
updateTextScale();

/* ========================= 8. 树顶 / 地面光晕 ========================= */
const glowTexture = makeRadialTexture([
  [0.00, 'rgba(255,255,252,1)'],
  [0.12, 'rgba(255,214,238,0.95)'],
  [0.35, 'rgba(255,158,208,0.40)'],
  [0.70, 'rgba(255,112,186,0.10)'],
  [1.00, 'rgba(255,92,172,0)'],
]);

const topGlow = new THREE.Sprite(new THREE.SpriteMaterial({
  map: glowTexture,
  color: 0xffd8ec,
  transparent: true,
  opacity: 0.95,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
}));
topGlow.position.set(0, CONFIG.tree.height * 0.5 + 0.4, 0);
topGlow.scale.set(7, 7, 1);
scene.add(topGlow);

const groundGlow = new THREE.Mesh(
  new THREE.PlaneGeometry(22, 22),
  new THREE.MeshBasicMaterial({
    map: glowTexture,
    color: 0xff9ac9,
    transparent: true,
    opacity: 0.5,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
);
groundGlow.rotation.x = -Math.PI / 2;
groundGlow.position.y = -CONFIG.tree.height * 0.5 - 0.2;
scene.add(groundGlow);

/* ========================= 9. 拖尾后处理 =========================
 * AfterimagePass 把上一帧以 damp 比例混入当前帧，运动粒子因此留下光轨。
 * damp 越大拖尾越长（0.82 是较柔和的值，可调到 0.92）。
 */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const afterimage = new AfterimagePass(0.82);
if (CONFIG.reduceMotion) afterimage.enabled = false;
composer.addPass(afterimage);

/* ========================= 10. 状态机与效果映射 ========================= */
const state = {
  morph: 0, morphTarget: 0,
  scatter: 0, scatterTarget: 0,
  jitter: 0, jitterTarget: 0,
  planetMorph: 0, planetTarget: 0,
  textIndex: 0,
  treeRotation: 0,
  planetSpin: 0,
};
let returnTimer = null;
let showTextTimer = null;

function setNumber(n) {
  dom.number.textContent = n;
  dom.number.classList.add('is-visible');
}
function hideNumber() { dom.number.classList.remove('is-visible'); }
function setLabel(text) { dom.label.textContent = text; }

function clearTimers() {
  clearTimeout(returnTimer);
  clearTimeout(showTextTimer);
}

/** 退出相册模式（供其它形态调用，确保互斥） */
function setFlipButtons(visible) {
  document.querySelectorAll('.flip-touch').forEach((b) => b.classList.toggle('is-hidden', !visible));
}
function leaveAlbum() {
  albumState.active = false;
  albumState.target = 0;
  albumState.flipping = false;
  if (dom.photoPage) dom.photoPage.classList.remove('is-visible');
  if (dom.albumTip) dom.albumTip.classList.remove('is-visible');
  setFlipButtons(false);
}

function goTree(jitter = 0, scatter = 0, scene = 'tree') {
  clearTimers();
  leaveAlbum();
  state.planetTarget = 0;
  state.morphTarget = 0;
  state.jitterTarget = jitter;
  state.scatterTarget = scatter;
  music.setScene(scene);
}

/** 2 指：粒子变形成带星环的自转星球 */
function goPlanet() {
  clearTimers();
  leaveAlbum();
  state.jitterTarget = 0;
  state.scatterTarget = 0;
  state.morphTarget = 0;
  state.planetTarget = 1;
  music.setScene('planet');
}

function showText(index) {
  clearTimers();
  leaveAlbum();
  state.planetTarget = 0;
  music.setScene('text');
  // ① 先完全散开
  state.morphTarget = 0;
  state.scatterTarget = 1.15;
  state.jitterTarget = 0;
  // ② 从散开状态重组成粒子文字
  showTextTimer = setTimeout(() => {
    state.textIndex = index;
    particleMaterial.uniforms.uTextIndex.value = index;
    state.morphTarget = 1;
    state.scatterTarget = 0;
  }, 520);
  // ③ 到时自动汇聚回圣诞树
  returnTimer = setTimeout(() => {
    state.morphTarget = 0;
    state.planetTarget = 0;
    state.scatterTarget = 0;
    music.setScene('tree');
    setLabel('✨ 已自动汇聚成圣诞树');
  }, CONFIG.autoReturnMs);
}

/* ---------- 礼物相册控制 ---------- */
/** 更新页码 HUD（角落显示，不遮挡 3D 主体） */
function updatePhotoPage() {
  if (dom.photoPage) dom.photoPage.textContent = `${albumState.photoIndex + 1} / ${PHOTO_COUNT}`;
}

/** 4 指：进入礼物相册模式（粒子先炸开，再收拢成彩屑场与相框） */
function enterAlbum() {
  clearTimers();
  clearTimeout(albumState.burstTimer);
  state.planetTarget = 0;
  state.morphTarget = 0;
  state.jitterTarget = 0;
  albumState.active = true;
  albumState.target = 1;
  albumState.flipping = false;
  albumState.flipPhase = 0;
  photoPlane.visible = true;
  photoPlane.material.map = photoTextures[albumState.photoIndex];
  photoPlane.material.opacity = 0;
  if (dom.photoPage) { updatePhotoPage(); dom.photoPage.classList.add('is-visible'); }
  if (dom.albumTip) dom.albumTip.classList.add('is-visible');
  setFlipButtons(true);
  setLabel('🎁 礼物相册 · 左右滑动翻页，握拳/1 指退出');
  music.setScene('album');
  // 炸开脉冲：粒子瞬间外扩，随后 album morph 收拢成新形态
  state.scatterTarget = CONFIG.album.burstStrength;
  albumState.burstTimer = setTimeout(() => { state.scatterTarget = 0; }, CONFIG.album.burstTime);
}

/** 退出相册：粒子炸开后重新汇聚回圣诞树，光球恢复挂饰 */
function exitAlbum() {
  clearTimeout(albumState.burstTimer);
  leaveAlbum();
  state.planetTarget = 0;
  state.morphTarget = 0;
  state.jitterTarget = 0;
  photoPlane.visible = false;
  frameGlow.visible = false;
  music.setScene('tree');
  state.scatterTarget = CONFIG.album.burstStrength;
  albumState.burstTimer = setTimeout(() => { state.scatterTarget = 0; }, CONFIG.album.burstTime);
}

/** 翻页：dir = 1 下一张，-1 上一张。单次触发由调用方节流。 */
function flipPage(dir) {
  if (!albumState.active || albumState.flipping || PHOTO_COUNT < 2) return;
  albumState.flipping = true;
  albumState.flipDir = dir;
  albumState.flipPhase = 0;
  // 提前算出目标索引，供动画中点切换纹理
  albumState.nextIndex = (albumState.photoIndex + dir + PHOTO_COUNT) % PHOTO_COUNT;
  music.duck(); // 翻页瞬间音乐轻微退让，随后自动恢复相册音量
}

/** 手势 → 视觉效果 的唯一入口（键盘降级也走这里） */
function applyGesture(gesture) {
  switch (gesture) {
    case 'ONE':
      setNumber('1');
      if (albumState.active) { exitAlbum(); setLabel('1 指 · 退出相册，回到圣诞树'); }
      else { setLabel('1 根手指 · 保持圣诞树形态'); goTree(0, 0); }
      break;
    case 'TWO':
      setNumber('2'); setLabel('2 根手指 · 带星环的自转星球 🪐'); goPlanet(); break;
    case 'THREE':
      setNumber('3'); setLabel('3 根手指 · 粒子逐渐散开'); goTree(0, 0.6, 'scatter'); break;
    case 'FOUR':
      setNumber('4');
      if (!albumState.active) { setLabel('4 根手指 · 打开礼物相册 🎁'); enterAlbum(); }
      else setLabel('已在相册 · 左右滑动翻页');
      break;
    case 'OK':
      hideNumber(); setLabel('OK · 张嘉雯圣诞快乐 🎄'); showText(0); break;
    case 'FIST':
      hideNumber();
      if (albumState.active) { exitAlbum(); setLabel('握拳 · 退出相册，回到圣诞树'); }
      else { setLabel('握拳 · 粒子重新汇聚'); goTree(0, 0); }
      break;
    case 'PALM':
      hideNumber(); setLabel('张掌 · 今天你开心了吗？'); showText(1); break;
    default:
      break;
  }
}

/* ========================= 11. 手势识别（MediaPipe HandLandmarker · WebGL/WASM） ========================= */
let handStream = null;      // 当前摄像头 MediaStream
let handposeInst = null;    // HandLandmarker 检测器（切换摄像头时复用）
let cameraFacing = 'user'; // 当前摄像头：user=前置 environment=后置
let detectionLoopId = 0;    // requestAnimationFrame 句柄
let detecting = false;      // 防止上一帧未推理完就叠加下一帧
let activeDelegate = 'GPU'; // 当前推理后端（GPU=WebGL / CPU=WASM）
let lastVideoTime = -1;     // 上一帧视频时间戳，用来判断画面是否真的在动
let videoMovingFrames = 0;  // 视频画面变化的帧数（诊断用）

/** 三维距离：MediaPipe 的 z 是相对深度，按配置降权后参与比例计算 */
function dist3(a, b) {
  const z = ((a.z || 0) - (b.z || 0)) * (CONFIG.gesture.zWeight ?? 1);
  return Math.hypot(a.x - b.x, a.y - b.y, z);
}

function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** 掌尺度：腕→中指根 与 食指根→小指根 取较大者。比例法用它归一化拇指/捏合距。 */
function handScale(lm) {
  return Math.max(dist3(lm[0], lm[9]), dist3(lm[5], lm[17]), 1e-4);
}

/** 手指伸展比：指根→指尖 除以 指根→第一指节。
 *  伸直时三节几乎共线 ≈ 2.4；弯曲时指尖缩回指根附近 ≈ 0.8。
 *  只用距离比，不受手掌旋转、远近、坐标系镜像影响，也不需要关节角。 */
function fingerRatio(lm, mcp, pip, tip) {
  const base = dist3(lm[mcp], lm[pip]);
  if (!(base > 1e-6)) return 0;
  return dist3(lm[mcp], lm[tip]) / base;
}

let lastGestureFlags = null; // 诊断用：食/中/无/小/拇 五根手指的伸直标志
let lastGestureDiag = null;  // 埋点日志：角度/拇指比/捏合比，显示在状态条供排查

/** 提取 6 维特征 [q0,q1,q2,q3,t,p]——与 tools/train_gesture.py 完全一致。
 *  q=四指伸展比，t=拇指张开比，p=捏合比，全部以掌尺度归一化，与手离镜头远近无关。 */
function extractFeatures(lm) {
  const scale = handScale(lm);
  const MCP = [5, 9, 13, 17], PIP = [6, 10, 14, 18], TIP = [8, 12, 16, 20];
  const q = MCP.map((mcp, i) => fingerRatio(lm, mcp, PIP[i], TIP[i]));
  const t = dist3(lm[4], lm[5]) / scale;
  const p = dist3(lm[4], lm[8]) / scale;
  return { feat: [...q, t, p], q, t, p };
}

/** 训练模型前向传播：标准化 → 线性层 → softmax，返回 {gesture, conf}。
 *  线性层权重 W 按 [class][feature] 排列，与导出的 gesture-model.js 一致。 */
function inferByModel(feat) {
  const m = GESTURE_MODEL;
  const c = m.classes.length;
  const z = feat.map((v, i) => (v - m.mean[i]) / m.std[i]);
  const logits = new Array(c);
  let maxL = -Infinity;
  for (let k = 0; k < c; k++) {
    let s = m.b[k];
    const row = m.W[k];
    for (let i = 0; i < z.length; i++) s += row[i] * z[i];
    logits[k] = s;
    if (s > maxL) maxL = s;
  }
  let sum = 0;
  const exps = logits.map(v => { const e = Math.exp(v - maxL); sum += e; return e; });
  let best = 0;
  for (let k = 1; k < c; k++) if (exps[k] > exps[best]) best = k;
  return { gesture: m.classes[best], conf: exps[best] / sum };
}

/** 规则法分类器（兜底）：模型不可用时用。四指伸展比数手指。 */
function classifyGestureRule(feat, q, t, p) {
  const g = CONFIG.gesture;
  const f = q.map(v => v > g.straightRatio);
  const thumb = t > g.thumbSpread;
  const n = f.filter(Boolean).length;
  if (p < g.okRatio && !f[0] && f[1] && f[2] && f[3]) return 'OK';
  if (n === 0) return 'FIST';
  if (n === 4) return thumb ? 'PALM' : 'FOUR';
  if (n === 3) return 'THREE';
  if (n === 2) return 'TWO';
  if (n === 1) return 'ONE';
  return null;
}

/** 手势分类 v29：训练模型为主（输出置信度），规则法兜底。
 *  返回 { gesture, conf }，conf 供触发逻辑做平滑判定。 */
function classifyGesture(lm) {
  const { feat, q, t, p } = extractFeatures(lm);
  lastGestureFlags = [q[0] > CONFIG.gesture.straightRatio, q[1] > CONFIG.gesture.straightRatio,
    q[2] > CONFIG.gesture.straightRatio, q[3] > CONFIG.gesture.straightRatio,
    t > CONFIG.gesture.thumbSpread];
  // 模型结构异常时降级到规则法（正常情况下不会发生，静态 import 保证模型可用）
  const modelOk = GESTURE_MODEL && Array.isArray(GESTURE_MODEL.classes) &&
    GESTURE_MODEL.W && GESTURE_MODEL.W.length === GESTURE_MODEL.classes.length;
  const infer = modelOk
    ? inferByModel(feat)
    : { gesture: classifyGestureRule(feat, q, t, p), conf: 1 };
  lastGestureDiag = {
    q: q.map(v => +v.toFixed(2)),
    t: +t.toFixed(2),
    p: +p.toFixed(2),
    c: +infer.conf.toFixed(2),
  };
  // 模型概率过低时视为"识别中"（不稳定状态，不触发）
  const gesture = infer.conf >= CONFIG.gesture.minConf ? infer.gesture : null;
  return { gesture, conf: infer.conf };
}

// 置信度平滑：对每类维护 EMA，最高且连续的那一类才触发。——比旧的"硬标签连续 N 帧"更快更稳。
const STABLE_FRAMES = CONFIG.gesture.triggerFrames;
let confEma = {};        // 每类置信度的指数移动平均
let candidate = null;
let candidateFrames = 0;
let activeGesture = null;
let lastTrigger = 0;
let handSeenOnce = false;   // 是否曾检测到手（用于诊断）
let handLostFrames = 0;     // 连续未检测到手的帧数
let procFrameCount = 0;     // 处理结果帧计数（诊断）
let lastResultAt = 0;       // 最后一次收到检测结果的时间（看门狗用）
let lastLmSignature = '';   // 上一帧骨架签名（检测"卡住的假手/静止画面"）
let frozenFrames = 0;       // 骨架连续不变的帧数
let hudPaused = false;      // 看门狗是否已提示"检测暂停"

/** 骨架合法性：HandLandmarker 自带置信度门槛已挡住误检，这里只挡真正退化的数值。
 *  尺度无关、结构宽松——不再用指节长度/腕点距离等容易误杀真手的硬阈值。 */
function checkHand(lm) {
  if (!lm || lm.length !== 21) return { ok: false, why: '空骨架' };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of lm) {
    if (!p || !isFinite(p.x) || !isFinite(p.y)) return { ok: false, why: 'NaN' };
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const span = Math.max(maxX - minX, maxY - minY);
  if (!(span > 0.01)) return { ok: false, why: '过小' };
  const scale = handScale(lm);
  if (!(span / scale > 1.2 && span / scale < 30)) return { ok: false, why: '比例异常' };
  // 腕到四个指根应大致同向同距（真实手掌是扇形），拦住错乱连线。
  const w = lm[0];
  const d = [5, 9, 13, 17].map(i => dist3(lm[i], w)).sort((a, b) => a - b);
  if (d[3] > 1e-6 && d[0] / d[3] < 0.45) return { ok: false, why: '掌骨异常' };
  // 中位数离群点：单点数值爆炸时中位数不受影响，能把它挑出来。
  const ds = [];
  for (let a = 0; a < 21; a++) for (let b = a + 1; b < 21; b++) ds.push(dist2(lm[a], lm[b]));
  ds.sort((x, y) => x - y);
  const med = ds[ds.length >> 1];
  if (!(med > 1e-6)) return { ok: false, why: '聚点' };
  const cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4;
  const cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4;
  for (const p of lm) if (dist2(p, { x: cx, y: cy }) > med * 15) return { ok: false, why: '离群点' };
  return { ok: true, scale, span };
}

/** 骨架签名：取关键指尖坐标（3 位小数），用于识别连续静止的假手/卡帧 */
function lmSignature(lm) {
  if (!lm) return '';
  return [0, 4, 8, 12, 16, 20].map(i => `${lm[i].x.toFixed(3)},${lm[i].y.toFixed(3)}`).join('|');
}

// 相册滑动翻页：记录食指指尖的原始归一化坐标轨迹
let swipeActive = false;
let swipeStartX = 0, swipeStartY = 0;
let lastFlipTime = 0;

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const pctx = dom.preview.getContext('2d');

/** 摄像头预览下方的小状态条：显示是否识别到手 / 手势，手机调试用 */
function updateDetectionHud(text, hasHand) {
  if (!dom.detectHud) return;
  if (dom.detectHud.textContent !== text) dom.detectHud.textContent = text;
  if (hasHand === undefined) hasHand = text.indexOf('已检测到手') === 0 || /^(1 指|2 指|3 指|4 指|OK|握拳|张掌|识别中)/.test(text);
  dom.detectHud.classList.toggle('has-hand', !!hasHand);
}

/** 画骨架：只画校验通过的干净骨架。MediaPipe 坐标是 [0,1] 归一化，
 *  用包围盒自适应铺满预览框（没手/垃圾数据在外层已拦，不会撑爆）。 */
function drawPreview(lm) {
  const w = dom.preview.width, h = dom.preview.height;
  pctx.clearRect(0, 0, w, h);
  if (!lm) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of lm) {
    if (!isFinite(p.x) || !isFinite(p.y)) { pctx.clearRect(0, 0, w, h); return; }
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const sx = Math.max(maxX - minX, 1e-4), sy = Math.max(maxY - minY, 1e-4);
  const s = Math.max(sx, sy);
  // 保持手型比例居中，x 镜像匹配 CSS 的 scaleX(-1)
  const px = (p) => w * (0.5 + (maxX - p.x - sx / 2) / s * 0.78);
  const py = (p) => h * (0.5 + (p.y - minY - sy / 2) / s * 0.78);

  pctx.strokeStyle = 'rgba(255,188,224,0.92)';
  pctx.lineWidth = 2;
  CONNECTIONS.forEach(([a, b]) => {
    pctx.beginPath();
    pctx.moveTo(px(lm[a]), py(lm[a]));
    pctx.lineTo(px(lm[b]), py(lm[b]));
    pctx.stroke();
  });
  pctx.fillStyle = '#ff7eb6';
  lm.forEach((p) => {
    pctx.beginPath();
    pctx.arc(px(p), py(p), 3, 0, Math.PI * 2);
    pctx.fill();
  });
}

function handleHandResults(results) {
  procFrameCount++;
  lastResultAt = performance.now();  // 看门狗：收到结果就刷新
  hudPaused = false;
  // MediaPipe HandLandmarker 返回：{ landmarks: [[21点]], worldLandmarks, handedness: [[{score,categoryName}]] }
  const rawLm = results && results.landmarks && results.landmarks[0];
  const hd = results && results.handedness && results.handedness[0] && results.handedness[0][0];
  const handScore = hd && typeof hd.score === 'number' ? hd.score : 0;

  // 区分两种情况：①没有输出骨架 ②输出了但被校验挡下（显示原因）
  // 埋点：v=视频尺寸 · r=readyState · mv=画面是否在动 · ct=视频时间 · d=后端 · n=返回手数
  const v = dom.video;
  if (v.currentTime !== lastVideoTime) { lastVideoTime = v.currentTime; videoMovingFrames++; }
  const moving = v.currentTime !== undefined && videoMovingFrames > 3 ? 'Y' : 'N';
  const handCount = (results && results.landmarks && results.landmarks.length) || 0;
  const vinfo = `v${v.videoWidth}x${v.videoHeight}r${v.readyState}mv${moving}t${(v.currentTime||0).toFixed(1)}d${activeDelegate}n${handCount}`;
  if (!rawLm) {
    pctx.clearRect(0, 0, dom.preview.width, dom.preview.height);
    candidate = null; candidateFrames = 0; activeGesture = null;
    confEma = {};
    swipeActive = false; lastGestureFlags = null; frozenFrames = 0;
    lastLmSignature = '';
    handLostFrames++;
    updateDetectionHud(`无骨架 ${vinfo} f${procFrameCount}`);
    // GPU/WebGL 在部分手机初始化成功但推理恒为空：连续 90 帧无手且画面在动 → 自动切 CPU
    if (activeDelegate === 'GPU' && videoMovingFrames > 5 && handLostFrames >= 90) {
      console.warn('[手势] GPU 后端连续 90 帧无结果，自动切换 CPU(WASM)');
      switchDelegate('CPU');
    }
    return;
  }
  const check = checkHand(rawLm);
  if (!check.ok) {
    pctx.clearRect(0, 0, dom.preview.width, dom.preview.height);
    candidate = null; candidateFrames = 0; activeGesture = null;
    confEma = {};
    swipeActive = false; lastGestureFlags = null; frozenFrames = 0;
    lastLmSignature = '';
    handLostFrames++;
    updateDetectionHud(`骨架异常:${check.why} ${vinfo} f${procFrameCount}`);
    return;
  }
  // handedness.score 是「左右手分类」置信度，不是「是否检测到手」——真手侧着/近镜头时
  // 经常低于 0.5，用它当门槛会误杀。landmarks 存在即代表检测器确认找到手，这里只挡极端值。
  // 注：检测到手的置信度已由 minHandDetectionConfidence/minHandPresenceConfidence 在引擎层把关。
  // 校验通过才画骨架（旧版在校验前画，垃圾坐标乱飞）
  drawPreview(rawLm);

  // 骨架签名只作诊断提示，不再否决检测：真实手势常常是"保持不动"的（握拳/OK/1 指），
  // 若连续几帧坐标不变就当假手 return，会导致"手不动就识别不到、一晃才闪现"。
  // 假手/卡帧已由 checkHand（聚点/NaN）+ activeGesture 去重（同手势只触发一次）覆盖。
  const sig = lmSignature(rawLm);
  if (sig === lastLmSignature) frozenFrames++;
  else { frozenFrames = 0; lastLmSignature = sig; }
  const stillNote = frozenFrames > 40 ? ' ·静止保持' : '';

  const lm = rawLm;
  handSeenOnce = true;
  handLostFrames = 0;

  // ===== 相册模式：水平滑动翻页（独立于手势分类，避免误判） =====
  if (albumState.active) {
    // 用食指指尖(8)的水平坐标，镜像画面里向右 = 原始 x 减小
    const hx = lm[8].x;
    const hy = lm[8].y;
    if (!swipeActive) {
      swipeStartX = hx; swipeStartY = hy; swipeActive = true;
    } else {
      const dx = hx - swipeStartX;   // >0 表示原始向右 = 镜像画面里向左
      const dy = hy - swipeStartY;
      const threshold = CONFIG.album.swipeRatio;
      // 水平位移超阈值、且水平主导（忽略垂直滑动），单次触发 + 冷却
      if (Math.abs(dx) > threshold && Math.abs(dx) > Math.abs(dy) * 1.5) {
        const now = performance.now();
        if (now - lastFlipTime > CONFIG.album.swipeCooldown && !albumState.flipping) {
          // 镜像画面：手向左滑(dx>0) → 下一张；向右滑(dx<0) → 上一张
          flipPage(dx > 0 ? 1 : -1);
          lastFlipTime = now;
        }
        swipeActive = false; // 触发后重置轨迹，防止一次滑动连续跳多张
      }
    }
  } else {
    swipeActive = false;
  }

  const { gesture, conf } = classifyGesture(lm);

  // 置信度 EMA：每类的概率做平滑，稳定跟手又抗单帧抖动。alpha 越大越灵敏。
  const alpha = CONFIG.gesture.emaAlpha;
  for (const cls of GESTURE_MODEL.classes) {
    const cur = gesture === cls ? conf : 0;
    confEma[cls] = confEma[cls] === undefined ? cur : confEma[cls] * (1 - alpha) + cur * alpha;
  }
  // 取平滑后概率最高且超过门槛的类
  let topCls = null, topVal = 0;
  for (const cls of GESTURE_MODEL.classes) {
    if (confEma[cls] > topVal) { topVal = confEma[cls]; topCls = cls; }
  }
  const stable = topVal >= CONFIG.gesture.minConf ? topCls : null;
  if (stable === candidate) candidateFrames++;
  else { candidate = stable; candidateFrames = 1; }

  // 实时显示识别到的手势 + 埋点诊断，手机截图即可定位问题
  const gName = stable ? ({ ONE:'1 指', TWO:'2 指', THREE:'3 指', FOUR:'4 指',
    OK:'OK', FIST:'握拳', PALM:'张掌' }[stable] || stable) : '识别中…';
  const bits = lastGestureFlags ? lastGestureFlags.map(b => b ? '1' : '0').join('') : '-----';
  // 埋点：q=四指伸展比 t=拇指张开比 p=捏合比 c=模型置信度 → 一眼看出误判原因
  const diag = lastGestureDiag
    ? ` q${lastGestureDiag.q.join('/')} t${lastGestureDiag.t} p${lastGestureDiag.p} c${lastGestureDiag.c}`
    : '';
  updateDetectionHud(`${gName} [${bits}]${diag} f${procFrameCount}`);

  if (candidateFrames >= STABLE_FRAMES && candidate && candidate !== activeGesture) {
    const now = performance.now();
    if (now - lastTrigger > CONFIG.gesture.triggerGap) {
      activeGesture = candidate;
      lastTrigger = now;
      applyGesture(activeGesture);
    }
  }
}

/** 懒加载 HandLandmarker：运行时(wasm)与模型都放在本地 vendor，避免 CDN/网络波动。
 *  优先 GPU(WebGL) 后端，失败自动回退 CPU(WASM)——两者都不依赖 WebGPU。 */
async function ensureHandpose() {
  if (handposeInst) return handposeInst;
  const base = new URL('../vendor/mediapipe/', import.meta.url).href;
  const vision = await FilesetResolver.forVisionTasks(base);
  const model = new URL('../vendor/mediapipe/hand_landmarker.task', import.meta.url).href;
  const make = (delegate) => HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: CONFIG.gesture.minHandScore,
    minHandPresenceConfidence: CONFIG.gesture.minHandScore,
    minTrackingConfidence: CONFIG.gesture.minHandScore,
  });
  try {
    handposeInst = await make(CONFIG.gesture.delegate);
    activeDelegate = CONFIG.gesture.delegate;
  } catch (err) {
    console.warn('[手势] WebGL 后端不可用，回退 CPU(WASM)', err);
    handposeInst = await make('CPU');
    activeDelegate = 'CPU';
  }
  return handposeInst;
}

/** 切换推理后端（GPU(WebGL) ↔ CPU(WASM)）：用于 GPU 静默失败时兜底。
 *  重建检测器（不碰摄像头流），重置无手计数，避免刚切过去就被自动降级逻辑再次触发。 */
async function switchDelegate(delegate) {
  if (delegate === activeDelegate) return;
  const base = new URL('../vendor/mediapipe/', import.meta.url).href;
  const vision = await FilesetResolver.forVisionTasks(base);
  const model = new URL('../vendor/mediapipe/hand_landmarker.task', import.meta.url).href;
  try {
    handposeInst = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: model, delegate },
      runningMode: 'VIDEO', numHands: 1,
      minHandDetectionConfidence: CONFIG.gesture.minHandScore,
      minHandPresenceConfidence: CONFIG.gesture.minHandScore,
      minTrackingConfidence: CONFIG.gesture.minHandScore,
    });
    activeDelegate = delegate;
    handLostFrames = 0;
    setLabel(`已切换到 ${delegate === 'GPU' ? 'WebGL' : 'CPU'} 推理后端，请再试一次手势`);
  } catch (err) {
    console.warn('[手势] 切换后端失败', err);
    setLabel('后端切换失败，可用键盘 1/2/3/4 体验');
  }
}

/** 自己管理摄像头流：HandLandmarker 不需要 Camera utils */
async function startCameraStream() {
  if (handStream) {
    handStream.getTracks().forEach(t => t.stop());
    handStream = null;
  }
  handStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: cameraFacing },
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
    audio: false,
  });
  dom.video.srcObject = handStream;
  await dom.video.play();
}

/** 逐帧推理循环：上一帧没跑完就跳过，避免推理队列堆积 */
function runDetectionLoop() {
  cancelAnimationFrame(detectionLoopId);
  const loop = async () => {
    if (!handposeInst || !handStream) return;
    if (!detecting && dom.video.readyState >= 2) {
      detecting = true;
      try {
        // detectForVideo 要求时间戳严格递增；用 performance.now() 天然满足
        const results = handposeInst.detectForVideo(dom.video, performance.now());
        handleHandResults(results);
      } catch (err) {
        console.warn('[手势] 帧处理异常（已跳过）', err);
        updateDetectionHud(`检测异常 f${procFrameCount}`);
      } finally {
        detecting = false;
      }
    }
    detectionLoopId = requestAnimationFrame(loop);
  };
  detectionLoopId = requestAnimationFrame(loop);
}

async function initHandTracking() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('当前环境不支持摄像头，可用键盘 1/2/3/4/O/F/P 体验');
  }
  await ensureHandpose();
  await startCameraStream();
  runDetectionLoop();
}

/** 切换前/后置摄像头：停旧流、换新流，检测器实例复用 */
async function switchCamera() {
  cameraFacing = cameraFacing === 'user' ? 'environment' : 'user';
  try {
    await startCameraStream();
    setLabel(`已切换到${cameraFacing === 'user' ? '前置' : '后置'}摄像头`);
  } catch (err) {
    console.warn('[手势] 切换摄像头失败', err);
    setLabel('摄像头切换失败，请刷新重试');
  }
}

/* ========================= 12. 动画循环 ========================= */
const clock = new THREE.Clock();

/**
 * 相册动画每帧更新：
 *  1) 光球"大粒子"在「树上挂点」与「相册漂浮」之间平滑插值；始终相机朝向，
 *     缓慢呼吸/漂浮；当前照片光球环绕相框高亮脉冲，其余缩小变淡。
 *  2) 翻页过渡：旧照片沿滑动方向模糊淡出移出，中点切换纹理，新片从反方向由虚到实滑入。
 *  3) 发光相框、照片平面透明度随相册形态淡入淡出。
 */
function updateAlbumAnimation(t, dt) {
  const m = albumState.morph; // 0=树上 1=相册（已平滑）
  const orbCfg = CONFIG.orb;

  // ---- 1) 光球大粒子：排布 / 漂浮 / 呼吸 / 高亮 / 相机朝向 ----
  giftOrbs.forEach((orb, i) => {
    const isCurrent = i === albumState.photoIndex;
    // 相册模式目标位置
    let albumTarget;
    if (isCurrent) {
      // 当前光球：环绕在竖版相框周围漂浮，脉冲呼吸
      const a = t * 0.9 + orb.phase;
      albumTarget = new THREE.Vector3(
        Math.cos(a) * (PHOTO_PLANE_W * 0.5 + 2.4),
        0.5 + Math.sin(a) * (PHOTO_PLANE_H * 0.5 + 0.4),
        4.4
      );
    } else {
      // 其余光球：缩小，分列相框两侧形成缩略环带
      const side = i < albumState.photoIndex ? -1 : 1;
      const rank = side < 0 ? (albumState.photoIndex - i) : (i - albumState.photoIndex);
      albumTarget = new THREE.Vector3(
        side * (PHOTO_PLANE_W * 0.5 + 2.8 + rank * 0.55),
        PHOTO_PLANE_H * 0.5 - rank * 1.7 + Math.sin(t * 1.2 + orb.phase) * 0.3,
        2.4
      );
    }

    // 树上位置加轻微上下漂浮
    const treeFloat = orb.treePos.clone();
    treeFloat.y += Math.sin(t * orbCfg.floatSpeed + orb.phase) * orbCfg.floatAmp;

    // 平滑插值到当前形态
    orb.current.lerpVectors(treeFloat, albumTarget, m);
    orb.group.position.copy(orb.current);

    // 始终正对相机：光球是一颗"面片粒子"，不是会露出侧面的实心球
    orb.mesh.quaternion.copy(camera.quaternion);
    // 轻微自转（绕视线轴），让封在里面的照片有微妙姿态变化
    orb.mesh.rotateZ(Math.sin(t * 0.4 + orb.phase) * 0.08 + orb.spin * 0.02);

    // 呼吸缩放 + 当前光球放大
    const breathe = 1 + Math.sin(t * 1.8 + orb.phase) * orbCfg.breatheAmp;
    const selectScale = 1 + m * (isCurrent ? (orbCfg.selectedScale - 1) : -0.25);
    // 面片 1×1 → 直径 = 2 × radius；这样 shader 里 r=1 恰好落在光球边缘
    const scale = orb.radius * 2 * breathe * selectScale;
    orb.mesh.scale.set(scale, scale, 1);

    // 材质 uniform：高亮脉冲 / 透明度 / 照片可见度（树上弱，相册强）
    const pulse = 0.5 + 0.5 * Math.sin(t * 4.2 + orb.phase);
    orb.mat.uniforms.uTime.value = t;
    orb.mat.uniforms.uHighlight.value = m * (isCurrent ? 0.6 + pulse * 0.4 : 0);
    orb.mat.uniforms.uAlbum.value = m;
    orb.mat.uniforms.uDim.value = (1 - m) * 1.0 + m * (isCurrent ? orbCfg.highlightGlow : orbCfg.dimOpacity);
    orb.mat.uniforms.uPhotoStrength.value =
      orbCfg.photoStrengthTree * (1 - m) + orbCfg.photoStrengthAlbum * m * (isCurrent ? 1 : 0.45);
  });

  // ---- 2) 翻页过渡动画：模糊淡出移出 → 换纹理 → 由虚到实滑入 ----
  if (albumState.flipping) {
    albumState.flipPhase += dt * CONFIG.album.flipDuration;
    const p = Math.min(1, albumState.flipPhase);
    const dir = albumState.flipDir;
    const smooth = (v) => v * v * (3 - 2 * v);
    if (p < 0.5) {
      // 前半段：旧片沿滑动方向移出，由清晰到模糊淡出
      const q = smooth(p * 2);
      photoPlane.position.x = -dir * PHOTO_PLANE_W * 1.0 * q;
      photoBlur.value = q;
      photoPlane.material.opacity = m * (1 - q) * 0.96;
    } else {
      // 后半段：新片从反方向滑入，由模糊到清晰
      const q = smooth((p - 0.5) * 2);
      photoPlane.position.x = dir * PHOTO_PLANE_W * 1.0 * (1 - q);
      photoBlur.value = 1 - q;
      photoPlane.material.opacity = m * q * 0.96;
    }
    if (p >= 0.5 && albumState.photoIndex !== albumState.nextIndex) {
      albumState.photoIndex = albumState.nextIndex;
      photoPlane.material.map = photoTextures[albumState.photoIndex];
      updatePhotoPage();
    }
    if (p >= 1) {
      albumState.flipping = false;
      albumState.flipPhase = 0;
      photoPlane.position.x = 0;
      photoBlur.value = 0;
    }
  } else {
    // 非翻页：照片居中清晰，随相册形态淡入
    photoPlane.position.x = 0;
    photoBlur.value = 0;
    photoPlane.material.opacity = m * 0.96;
  }
  // 照片轻微上下浮动 + 发光边框跟随
  photoPlane.position.y = 0.5 + Math.sin(t * 1.1) * 0.12;
  photoPlane.rotation.y = Math.sin(t * 0.5) * 0.03;
  frameGlow.visible = photoPlane.visible;
  frameGlow.position.x = photoPlane.position.x;
  frameGlow.position.y = photoPlane.position.y;
  frameGlow.rotation.y = photoPlane.rotation.y;
  frameGlow.material.opacity = CONFIG.album.frameGlow * photoPlane.material.opacity;
}
let lastT = 0;

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  const dt = Math.min(t - lastT, 0.05);
  lastT = t;

  // 手势回调看门狗：超过 3 秒没有任何检测结果 → 明确提示，不再停留在旧状态
  if (lastResultAt > 0 && !hudPaused && performance.now() - lastResultAt > 3000) {
    hudPaused = true;
    candidate = null; candidateFrames = 0; activeGesture = null;
    confEma = {}; lastGestureFlags = null;
    updateDetectionHud('检测已暂停 · 请刷新页面');
  }

  // 所有形态参数做指数平滑，过渡自然
  state.morph += (state.morphTarget - state.morph) * 0.045;
  state.scatter += (state.scatterTarget - state.scatter) * 0.05;
  state.jitter += (state.jitterTarget - state.jitter) * 0.09;
  state.planetMorph += (state.planetTarget - state.planetMorph) * 0.05;
  albumState.morph += (albumState.target - albumState.morph) * 0.05;

  const speed = CONFIG.reduceMotion ? 0.15 : 1;
  state.treeRotation += dt * 0.13 * speed;            // 树缓慢自转
  state.planetSpin += dt * 0.5 * speed;               // 星球自转（星环在 shader 里更快）

  // ===== 相册动画：光团排布 + 翻页过渡 + 照片平面 =====
  updateAlbumAnimation(t, dt);

  const u = particleMaterial.uniforms;
  u.uTime.value = t;
  u.uMorph.value = state.morph;
  u.uScatter.value = state.scatter;
  u.uJitter.value = state.jitter;
  u.uTextIndex.value = state.textIndex;
  u.uTreeRotation.value = state.treeRotation;
  u.uPlanetMorph.value = state.planetMorph;
  u.uPlanetSpin.value = state.planetSpin;
  u.uAlbumMorph.value = albumState.morph;
  starMat.uniforms.uTime.value = t;
  // 彩屑粒子场：时间流动 + 随相册形态淡入淡出
  confettiMat.uniforms.uTime.value = t;
  confettiMat.uniforms.uReveal.value = albumState.morph;

  // 光晕随形态呼吸：变成文字/星球/相册时树顶光晕淡出
  topGlow.material.opacity = 0.95 * (1 - state.morph * 0.65) * (1 - state.planetMorph)
    * (1 - albumState.morph) * (1 - Math.min(state.scatter, 1) * 0.4);
  topGlow.scale.setScalar(7 + Math.sin(t * 1.6) * 0.5);
  groundGlow.material.opacity = 0.42 + 0.12 * Math.sin(t * 1.1);

  // 相机轻微浮动
  if (!CONFIG.reduceMotion) {
    camera.position.x = Math.sin(t * 0.18) * 0.45;
    camera.position.y = 1.2 + Math.cos(t * 0.22) * 0.28;
    camera.lookAt(0, 0, 0);
  }

  composer.render();
}
animate();
setLabel('✨ 3D 场景就绪 · v30 · 开启摄像头或按键体验');

/* ========================= 13. 事件绑定 ========================= */
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  const pr = renderer.getPixelRatio();
  particleMaterial.uniforms.uPixelRatio.value = pr;
  starMat.uniforms.uPixelRatio.value = pr;
  updateTextScale(); // 窗口尺寸变化后重新计算文字适配宽度
});

// 音乐开关：这是浏览器认可的真实用户手势，用来解锁音频上下文
if (dom.musicBtn) {
  dom.musicBtn.addEventListener('click', async (event) => {
    event.stopPropagation();
    await music.toggle();
  });
}

// 触摸按钮栏、相册翻页按钮、预览内切换按钮已移除：
// 手势识别已稳定，键盘 1/2/3/4/O/F/P + ←→ 翻页 为无摄像头时的降级方式。

// 摄像头/后端切换仍保留代码入口（元素若存在则可用，便于调试时临时恢复预览）
if (dom.cameraSwitch) {
  dom.cameraSwitch.addEventListener('click', (event) => {
    event.stopPropagation();
    switchCamera();
  });
}
// 推理后端切换：识别不到手时点 ⚡ 在 WebGL / CPU 之间切换（GPU 静默失败时尤其有用）
if (dom.delegateSwitch) {
  dom.delegateSwitch.addEventListener('click', (event) => {
    event.stopPropagation();
    switchDelegate(activeDelegate === 'GPU' ? 'CPU' : 'GPU');
  });
}

document.getElementById('start-btn').addEventListener('click', async (event) => {
  // ★ 音乐联动：在用户手势栈内同步解锁播放，之后切场景才能听到音量联动变化
  music.ensurePlaying();
  const btn = event.currentTarget;
  btn.disabled = true;
  btn.textContent = '正在开启…';
  try {
    await initHandTracking();
    dom.panel.classList.add('is-hidden');
    // 预览已视觉隐藏（仅 video 出帧供识别），无需 remove is-hidden
    setLabel('摄像头已就绪 · 伸手试试手势 ✋');
    // 若 6 秒都没识别到手，给出排查提示（预览已隐藏，提示改为通用引导）
    setTimeout(() => {
      if (!handSeenOnce) setLabel('⚠ 没识别到手：让手正对摄像头、光线充足，再试一次');
    }, 6000);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = '重新开启摄像头';
    const hint = /permission|denied|not allowed/i.test(err.message || '')
      ? '请在浏览器弹窗中允许摄像头权限，并通过 HTTPS 访问。'
      : '可尝试刷新，或用键盘 1/2/3/4/O/F/P 体验全部效果。';
    dom.note.textContent = `开启失败：${err.message || err}。${hint}`;
    dom.note.classList.add('is-error');
  }
});

// 无摄像头时的键盘降级：1/2/3/4/O/F/P 对应七种手势；相册内左右方向键翻页
window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  // M 键：音乐开关（键盘降级同样需要一次真实按键来解锁音频）
  if (key === 'm') { music.toggle(); return; }
  // 相册内方向键翻页（走与手势相同的翻页函数，含防抖）
  if (albumState.active && (key === 'arrowleft' || key === 'arrowright')) {
    const now = performance.now();
    if (now - lastFlipTime > CONFIG.album.swipeCooldown && !albumState.flipping) {
      flipPage(key === 'arrowright' ? 1 : -1); // → 下一张，← 上一张
      lastFlipTime = now;
    }
    if (!dom.panel.classList.contains('is-hidden')) dom.panel.classList.add('is-hidden');
    return;
  }
  const map = { '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR', 'o': 'OK', 'f': 'FIST', 'p': 'PALM' };
  const gesture = map[key];
  if (!gesture) return;
  applyGesture(gesture);
  if (!dom.panel.classList.contains('is-hidden')) dom.panel.classList.add('is-hidden');
});
