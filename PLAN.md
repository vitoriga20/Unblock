# 遮不住 Unblock：技术验证执行计划

## 目标

输入同一块平面（黑板或投影幕）的两张照片：不同视角、遮挡位置不同。输出两张以基准视角看、遮挡区域由另一张的真实可见像素填补、并经过正视校正的完整板书图。

核心命题是"**用另一张照片的真实像素救回被挡住的内容**"，不是 AI 生成。

## 非目标

- 不做 App、不做 UI、不做实时
- 不做生成式补全（inpainting 一律不进入主链路）
- 不做多张（3 张以上）融合，本次固定两张
- 不追求教科书级无缝，先保证"文字能读"

## 输入约定

- 目录：`有遮挡的图片/`
- 命名：`{group}_{idx}.jpg`，`idx=0` 是基准视角（最终输出以它为构图）
- 每组两张，必须是同一次演示内容（板书未变），否则帧差法失效
- 四角允许人工覆盖：自动检测失败时，在配置里写归一化四角坐标

## 处理链路

1. 读图：中文路径安全读取（`np.fromfile` + `imdecode`）
2. 屏幕四角定位：暗区分割 / 直线交点双策略，输出候选四边形
3. 对齐：`aux -> base` 单应矩阵（先四角粗对齐，再用 SIFT+RANSAC 在重叠区精修）
4. 遮挡 mask：warp 后帧差 + 自适应阈值 + 开闭运算 + 连通域面积过滤
5. 可见性判定：`base` 的遮挡区 ∩ `aux` 可见区 = 可补区；两者都遮 = 盲区
6. 增益补偿：重叠区逐通道增益估计，消除曝光/白平衡差
7. 像素搬运：只在 `base` mask 内从 warp 后的 `aux` 采样，其余保持原样
8. 融合：距离变换羽化边界
9. 正视校正：`base` 四角 -> 目标矩形
10. 指标与诊断图：覆盖率、盲区率、重投影误差、接缝跳变

## 分阶段验收（失败不往下走）

| 阶段 | 产物 | 通过判据 |
|---|---|---|
| S1 四角 | 屏幕四边形叠图 | 四条边贴合屏幕边框，肉眼确认 |
| S2 对齐 | aux warp 后与 base 的叠图 | 文字重影小于约 2px |
| S3 mask | mask 叠加原图 | 覆盖人物与后脑勺，不误伤文字 |
| S4 合成 | 合成图 + 接缝局部放大 | 无硬接缝、无色块 |
| S5 正视 | 矩形输出 | 文字可读 |

## 指标

- 覆盖率：base 遮挡像素中被 aux 真实可见像素填补的比例（客观）
- 盲区率：两张都不可见的比例（物理上限，不能优化，只能报告）
- 重投影误差：重叠区 warp 后残差
- 接缝跳变：mask 边界内外梯度差
- 人工可读打分：1-5

## 已知风险

- 两张都挡住的地方必然无法恢复，这是物理上限
- 屏幕边框被遮挡时四角检测会失败，需人工覆盖
- 屏幕反光、灯光造成亮度不均，会污染帧差，必须先补偿
- 视角差大时重采样会糊，文字可读性下降

## 运行方式

```powershell
python -m unblock.annotate --config config.json          # 标注缺失的四角（点击 4 下，s 保存）
python -m unblock.annotate --config config.json --render   # 只渲染现有四角核对
python -m unblock.pipeline --config config.json --stage all
```

## 迭代记录

### 阶段一：四角交互标注
自动四角检测会被 PPT 内部卡片边框干扰，结论是不可靠。改为人工点击标注：
鼠标左键点四个角，`s` 保存并直接写回 `config.json`，`r` 重来，`q` 跳过。
标注结果按归一化到原图坐标存储，换分辨率也不影响。

### 阶段二：小目标分割
只露后脑勺的观众召回率低，导致 aux 侧漏判，把另一张的观众贴进结果。
改为分块推理（`segment.tile=2`，20% 重叠）+ 降低置信度，召回明显提升。
底部分块边缘按格切分，避免重叠区重复检出造成块状伪影。

### 阶段三：局部非刚性精修
全局单应之后仍有局部残余位移（镜头畸变、幕布不绝对平），表现为文字轻微重影。
在已对齐的 aux 之上再匹配一次特征，把残差网格化（`residual_grid=6`）取中位数、
平滑后插值成稠密位移场，最后 remap 一次。位移超过 `residual_max_px` 的点直接丢弃，
防止误匹配把画面拉变形。

### M1：Android 端移植（真机验收）

真机：一加 Ace 5（PKG110），Android 15，arm64-v8a。技术栈：Kotlin + OpenCV 4.10.0（Maven）+ ONNX Runtime 1.19.2，无 NDK。

**adb 接通（Windows）**：ADB 接口被绑到微软通用 WinUSB 驱动，缺 Android 的接口 GUID，
`adb devices` 全程空表。需要以管理员权限给该接口的 `Device Parameters` 加
`DeviceInterfaceGUIDs = {f72fe0d4-cbcb-407d-8814-9ed673d0dd6b}`（REG_MULTI_SZ），再禁用/启用该设备。
值绑在具体设备实例上，换 USB 口或换机会丢；长期稳定需装通用 ADB INF。

**端上踩到的三个坑**：

1. `gain` 阶段整块发黑。增益图用 `base/aux` 逐像素相除，aux 在变换后画面外是纯黑，
   除零产生 inf/NaN，`inf×0=NaN` 又被大核 blur 扩散到全图，`applyGain` 后 `convertTo` 把 NaN 压成 0。
   修法：分母下限取 1；平滑后 `den` 再加下限；无样本区直接置 1。实测 gain 均值 0.97 / 0.92。
2. 遮挡人物没被填。YOLO 的 mask 从 160×160 proto 栅格映射回原图时漏了 letterbox padding 偏移，
   mask 整体下移约 500px（4096 宽图）。修法：`proto = (x*scale + pad) / 4`。
3. 对齐整体错开。Android 原来用四角单应，而 PC 端实际生效的是 SIFT 特征对齐。同一组图上
   两种单应差 577px / 278px，屏幕区灰度差中位数 16 → 8。修法：移植 SIFT（1280 长边检测、
   ratio 0.75、RANSAC 6.0、≥30 内点）+ 四角合理性兜底，特征不足时退回四角单应。

**实测（两组合成结果与 PC 端一致，文字完整）**：

| 组 | 总耗时 | 覆盖 | 盲区 | 接缝比 |
|---|---|---|---|---|
| group0 | 4.09 s | 100% | 0% | 3.16 |
| group1 | 4.01 s | 100% | 0% | 1.96 |

分段（group0 / group1）：decode 0.25 / 0.34s、SIFT 0.45 / 0.37s、warp 0.04 / 0.31s、
分割 0.62 / 1.14s、遮挡 0.39 / 0.40s、gain 0.39 / 0.29s、blend 1.44 / 0.58s、rectify 0.43 / 0.53s。

**提速手段**（画面与提速前逐项指标一致，仅耗时变化）：
增益图本身是低频的，改在 1/4 分辨率上估计再双线性上采样，平滑核按比例缩到 11，
这一项从 3.4s 降到 0.3~0.4s；SIFT 工作分辨率 1280 → 1024（内点数仍远高于阈值）；
正视校正 INTER_CUBIC → INTER_LINEAR。

**遗留**：当前最大单项是 blend（group0 1.44s），它是在全分辨率上做三次浮点乘加，
如需再压可在遮挡 mask 的外接框内做，其余区域直接取 base；APK 106MB（双 ABI），后续只留 arm64-v8a 可降到约 60MB。

**构建/安装/运行闭环**：

```powershell
$env:JAVA_HOME='D:\KB\Unblock\.toolchain\jdk\jdk-17.0.20.1+1'
$env:ANDROID_HOME='D:\KB\Unblock\.toolchain\android-sdk'
& 'D:\KB\Unblock\.toolchain\gradle\gradle-8.9\bin\gradle.bat' assembleDebug --offline --rerun-tasks --console=plain
$adb='D:\KB\Unblock\.toolchain\android-sdk\platform-tools\adb.exe'
& $adb install -r 'D:\KB\Unblock\android\app\build-app\outputs\apk\debug\app-debug.apk'
```

注意两点：Gradle 会误报 `UP-TO-DATE` 不重编，改代码后一律加 `--rerun-tasks`；
调试中间图由 `PipelineConfig.dumpDebug` 控制，默认关闭，开启后落到
`/sdcard/Android/data/com.unblock.app/files/debug/`。

### M2：拍摄交互

顺序：四角粗定位与拖拽 -> 拍照/选图与多图融合 -> 出图留存。

已完成：
- 屏幕四角自动粗定位移植到端上（LSD 线段 → 方向聚类 → 交点求四边形 → 约束筛选）。
  端上 group0 与人工标注平均偏差 422.7px，与 PC 端同一算法的 423px 一致；group1 两端都检不出。
  结论：这个算法只能当“粗初值”，检测失败或偏差大时退回内缩矩形，由用户拖动。
- 四角拖拽编辑界面（`QuadEditorView` + `QuadEditorActivity`）：图片上四个手柄可拖、
  可重新检测、可重置、可确认回传。已在真机上验证「打开 → 自动初值 → 拖动 → 确认 → 回传」闭环。
- 输入链路（`PhotoStore`）：相册多选 2~4 张（Photo Picker，无需存储权限），统一按 EXIF 摆正、
  长边 2400 落成缓存的 JPEG，四角编辑与合成共用同一份，避免分辨率/方向不一致。
- 多图融合（`Pipeline.runMulti`）：底图 + 若干补图逐张迭代，每轮对“当前合成图”重新做人物分割，
  已补好的区域不会再被当成遮挡；用户只需拖动一次底图四角，其余照片的四角由其单应推算。
  实测 3 张（底图 + 2 张补图）总耗时 4.0s，覆盖 100%、盲区 0%。

- 出图留存：合成结果自动存入 App 内的最近任务（`TaskStore`，私有目录 + 同名 json 记元信息），
  「最近任务」列表可回看，结果页另提供「保存到相册」（`GallerySaver`，API 29+ 走 MediaStore 写 Pictures/Unblock）。
  真机验证：合成 3 张 → 自动入库 → 列表点开 → 保存到相册，文件落在 `/sdcard/Pictures/Unblock/`。

待办：相机直拍入口；Android 9 及以下的相册写入还需要存储权限申请。

### M3：移除人物后的轮廓修复

**现象**：人物被补掉之后，原位置周围仍有一圈可见轮廓。用真机缓存的用户素材（黑板讲课，1390×632）放大核对，
轮廓不是几何重影，而是**人物投在板面上的半影**留在填充区之外。

**四个成因与对应改法**：

1. 填充区只覆盖人形轮廓，半影在圈外 → `occlusion._shadow_band`：以 `base/aux` 灰度比相对全局中位增益的偏暗量判定阴影，
   只取 base 偏暗方向（aux 偏暗说明另一张才是被挡的），并用**距离场**限制在人物 `shadow_max_ratio × 长边` 范围内。
   用膨胀核做这个范围会退化成 O(k²) 的巨型核，距离场是线性开销。
2. 边界近乎硬替换（`edge_soft_px=2`、`min_alpha=0.98`）→ `compose.blend` 改成**距离场线性羽化**：
   边界 alpha 从 `min_alpha` 平滑升到 1，羽化带落在已外扩的干净背景上，不会把遮挡物混回来。
3. 增益图跨过接缝 → `gain_exclude_px`，接缝外侧一圈不参与 `base/aux` 比值统计，避免人物残留把整张增益拉偏。
4. 端上缺 `refine_residual` → Android 移植 `Geometry.refineResidual`（SIFT → 网格 × 6 位移均值 → 空格邻域补齐 →
   平滑 → 双线性上采样 → remap），`residual_max_side=1280`，实测每张 +0.8s。

**参数单一源**：新增参数全部进 `config.json`；Android 侧新增 `ConfigLoader`，从 `assets/config.json` 的 `params` 段读同一套键，
代码里只保留 data class 缺省值。`compose.seam_band_px`、`debug.dump` 也走同一份配置。

**实测（真机 一加 Ace 5）**：

| 组 | 接缝比 改前 → 改后 | 接缝跳变（灰度级） | 残差精修 | 总耗时 |
|---|---|---|---|---|
| group0 | 3.16 → 2.91 | 20.86 | on（1312 点，17.7px） | 4.04s → 5.48s |
| group1 | 1.96 → 1.90 | 31.78 | on（771 点，11.1px） | 3.39s → 4.65s |

PC 端同参数：group0 接缝比 2.512 → 2.445、跳变 21.01；group1 1.830 → 1.844、跳变 10.61。
用户素材 A/B（同一四角、只切 `shadow_enable`）：关闭时人物原位残留清晰暗影，开启后消失。

**注意**：`seam_step` 在板书场景会被穿过边界的粉笔字推高（group1 31.78 主要来自文字边缘），
判断轮廓问题要看同视角局部放大图，不能只看这一个数字。

## 工程约定

- 所有可调参数进 `config.json`，业务代码不写字面量
- 每个阶段保留 debug 产物，便于归因
- 新增一组：图片放进输入目录，在 `config.json` 的 `groups` 加一条即可
