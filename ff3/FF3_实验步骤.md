# FF3 植入缺陷实验：操作步骤

目标：衡量 Auto-Screen 的两种比对方法（像素比对 / 灰度比对）在**版本对比**中发现本地化显示错误的能力，指标是 Recall 和 Precision。对应第 1 章的 FF3。

预计用时：植入缺陷 3–4 小时，构建和运行 1–2 小时，标注 20–30 分钟，评估 10 分钟。

---

## 0. 实验设计（写论文时要用）

| 项目 | 设定 |
|---|---|
| 被测 App | KatApp（Flutter），iPhone 18 Pro 模拟器，iOS 27 |
| Build A | 原版（你已经加好简体和繁体中文的版本） |
| Build B | 在 A 的基础上植入 14 个缺陷：Gekürzter Text 4、Beschnittener Text 3、Überlappung 3、Verdrängtes Element 4（D07、D12 未实施，见 Protokoll） |
| 缺陷只放在 | **德语版**（baseline_DE_DE）。EN、ZH-CN、ZH-TW 的页面作为"无缺陷"对照 |
| 页面 | katapp.json 里的 10 个页面 × 4 种语言 = 40 组页面比对 |
| 运行 | A1、A2（A 跑两次）、B1。可选：B2，用来检查稳定性 |
| 比对 | B1 对 A1 → Recall、Precision。A2 对 A1 → 纯噪声带来的误报 |
| 判定"发现" | 至少有一个比对框与缺陷区域重叠 |

缺陷类别和第 2 章表 2.1 保持一致：**Gekürzter Text / Beschnittener Text / Überlappung / Verdrängtes Element**。

---

## 1. 植入缺陷（Build B）

原则：**由德语文本变长来触发缺陷**，代码里只加"潜在的"尺寸限制。这些限制在英文、中文这种文本较短的情况下不应该改变外观。这样模拟的是现实里常见的情况："布局写死了，翻译一长就出问题"。

建议先 `git checkout -b seeded-defects`，每加一个缺陷就 commit 一次，commit 信息写缺陷编号（D01…）。这些 commit 本身就是你的实验记录。

### 四类缺陷的做法（Flutter）

**A. Gekürzter Text（D01–D04）：横向截断，出现省略号**
- 给某个文本加上 `maxLines: 1, overflow: TextOverflow.ellipsis`，放进 `SizedBox(width: …)` 或 `ConstrainedBox(maxWidth: …)`，宽度设成英文刚好放得下。
- 在 `app_de.arb` 里把对应 key 的德语文本加长，比如用长的复合词。

**B. Beschnittener Text（D05、D06、D08）：纵向被裁切**
- 文本放进 `SizedBox(height: 一行的高度)` 和 `ClipRect`，然后 `maxLines: 2` 或不设限制，德语文本加长到需要换行，第二行就会被切掉。
- 注意：要的是**被切掉**，不是出现省略号。

**C. Überlappung（D09–D11）：元素重叠**
- 用 `Stack` 和 `Positioned` 把两个元素放在固定位置。德语文本一变长，就会压到旁边的图标或按钮上。
- 或者在一行里放 `Text` + 图标，文本用 `softWrap: false, overflow: TextOverflow.visible`，图标用 `Positioned` 固定在右侧。

**D. Verdrängtes Element（D13–D16）：元素被挤出可见区域**
- 把一行内容放进 `SingleChildScrollView(scrollDirection: Axis.horizontal, physics: NeverScrollableScrollPhysics())`。德语文本变长后，右边的按钮会被推到屏幕外面。

### ⚠️ 关键注意事项

1. **不要触发 Flutter 的溢出警示条。** 模拟器上只能跑 debug 构建，`Row`/`Column` 一旦溢出（RenderFlex overflow），就会画出黄黑相间的斜条纹。这样任何比对方法都能轻松发现，实验就不真实了。上面四种做法都是为了避开这种情况。每次截图后检查一遍：**画面里不能出现黄黑条纹**，出现了就换一种写法。
2. **对照语言必须保持不变。** 构建 B 之后，先看一眼 EN、ZH-CN、ZH-TW 的页面，应该和 A 一模一样。如果因为加了尺寸限制，英文布局也变了，就调整宽度。否则这些变化会被算成误报，但其实是你自己改出来的。
3. **14 个缺陷尽量分散在不同页面**，每个页面最多 3 个。少数页面保持没有缺陷，这样德语里也有无缺陷的页面作为对照。
4. 缺陷要**看得见，但也要贴近现实**：截掉一两个字母、压住一点点图标，都算有效。不要做成整块消失那种极端情况。

---

## 2. 构建和运行

```bash
# Build A
git checkout main && flutter build ios --simulator --debug
# 运行 Auto-Screen（和平时一样，模式选 DG 或 SoM 都行），把 screenshots 文件夹整个复制出来：
cp -R <Auto-Screen>/Qwen_trained/screenshots runs/A1/screenshots
# 再跑一次 A，复制到 runs/A2/screenshots

# Build B
git checkout seeded-defects && flutter build ios --simulator --debug
# 运行 Auto-Screen，复制到 runs/B1/screenshots
```

- 三次运行使用**同一台模拟器、同一个 iOS 版本、同样的系统设置**（时间、深色模式、字号）。
- 如果某个页面在某次运行中没有截到，评估脚本会列出来。缺页的情况要在论文里写明。

---

## 3. 标注缺陷位置（Ground Truth）

1. 复制 `seeded_defects_template.json` 为 `seeded_defects.json`，填满 14 条。
2. 每条的 `bbox` 用下面的命令获取：用鼠标框出缺陷区域，按回车，把输出的坐标填进去。

```bash
pip install opencv-python   # 第一次需要（这里要的是带界面的版本）
python3 bbox_picker.py runs/B1/screenshots/baseline_DE_DE/settings_page.png
```

框的范围：**刚好包住受影响的元素**，比如整个被截断的文本，或者重叠的两个元素。不要框得太大，否则会把附近无关的比对框也算成"发现"。

---

## 4. 评估

```bash
# 缺陷检测
python3 ff3_eval.py --tool-dir ~/Auto-Screen \
  --baseline runs/A1/screenshots --actual runs/B1/screenshots \
  --gt seeded_defects.json --out results/B1_vs_A1

# 噪声对照（不需要 --gt）
python3 ff3_eval.py --tool-dir ~/Auto-Screen \
  --baseline runs/A1/screenshots --actual runs/A2/screenshots \
  --out results/A2_vs_A1
```

脚本直接调用你的 `compare_images.py`，**不改动它**，所以统计的就是 HTML 报告里显示的那些框。输出：
- `summary.json`：每种方法的 Recall（总体和分类别）、框级 Precision、页面级结果、漏掉的缺陷编号
- `page_results.csv`：每个页面的明细
- `diffs/`：比对图，可以挑几张放进论文

把 `results/` 整个文件夹发给我，我来写第 5.x 节。

---

## 5. 论文里要如实写的局限
- 缺陷是人为植入的，数量为 14 个，只在一个 App、一种设备上测试。
- 缺陷的位置和类型是作者自己选的，可能偏向比较容易发现的情况。
- 只测了德语版本里的缺陷。
- 只能在 debug 构建中测试，这是 Flutter 在 iOS 模拟器上的限制。
