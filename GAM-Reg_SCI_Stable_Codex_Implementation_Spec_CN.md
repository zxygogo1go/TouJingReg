# GAM-Reg（SCI 稳妥版）Codex 实现规格

> **用途**：本文件是可直接交给 Codex 的工程实现说明。请按本文定义的坐标约定、模块接口、张量形状、前向流程、损失和测试逐项实现，不要自行改变变换方向或省略关键模块。
>
> **模型名称**：GAM-Reg（Gaussian Anatomy Matching Registration）
>
> **任务**：三维头颈部可变形图像配准；默认 moving image 配准到 fixed image。支持 CT–CT，经过模态稳健特征处理后也可用于 CT–CBCT。
>
> **第一版目标**：先获得可复现、稳定、结构完整的 SCI 版本，不加入 diffusion、RL、LLM、3DGS rendering、纵向时序或复杂 source–sink 模型。

---

## 0. Codex 必须遵守的实现原则

1. 使用 PyTorch 实现，所有模块均继承 `torch.nn.Module`。
2. 输入张量统一采用 `[B, C, D, H, W]`。
3. `grid_sample` 的坐标顺序统一为 `(x, y, z)`，范围为 `[-1, 1]`；禁止在不同模块中混用 `(z, y, x)`。
4. moving 和 fixed 必须先重采样到相同 spacing、相同方向和相同体积尺寸；第一版假设已有刚性或仿射粗配准。
5. Gaussian 中心、尺度、协方差和位移全部在归一化坐标系 `[-1, 1]^3` 中表示。
6. Gaussian 2-Wasserstein 距离中的 SPD 矩阵平方根必须在关闭 AMP 的 `float32` 环境中计算，并对特征值做下界截断。
7. Sinkhorn 必须使用 log-domain 实现，禁止直接反复归一化 `exp(-C/epsilon)`，避免数值下溢。
8. velocity head 必须零初始化，使训练初始状态接近 identity registration。
9. 模型必须同时返回 forward map 和 inverse map；禁止仅返回一个含义不明确的 flow。
10. 最终代码必须包含单元测试，确保变换方向、Sinkhorn 边缘分布、Gaussian W2、scaling-and-squaring 和梯度传播正确。

---

# 1. 问题定义与变换方向

给定：

- moving volume：`I_m`
- fixed volume：`I_f`

模型预测 stationary velocity field：

\[
v:\Omega\rightarrow\mathbb R^3.
\]

通过 scaling-and-squaring 得到：

\[
\phi_{m\rightarrow f}=\exp(v),
\qquad
\phi_{f\rightarrow m}=\exp(-v).
\]

其中：

- `phi_fwd = phi_m2f`：moving 坐标映射到 fixed 坐标；
- `phi_inv = phi_f2m`：fixed 坐标映射到 moving 坐标。

将 moving 图像重采样到 fixed 网格时，必须使用 inverse map：

\[
I_{m\rightarrow f}(x_f)
=
I_m\big(\phi_{f\rightarrow m}(x_f)\big).
\]

代码约定：

```python
warped_moving = spatial_transform(moving, phi_inv)
```

Gaussian matching 产生的是 moving token 到 fixed token 的对应，因此 token displacement 为：

\[
d_i=\hat\mu_i^f-\mu_i^m.
\]

最终 forward map 必须满足：

\[
\phi_{m\rightarrow f}(\mu_i^m)\approx\hat\mu_i^f.
\]

---

# 2. 总体架构

```text
Moving Volume ──┐
                ├── Shared Multi-scale 3D Encoder
Fixed Volume  ──┘
                         │
                         ├── moving features {Fm0, Fm1, Fm2, Fm3}
                         └── fixed  features {Ff0, Ff1, Ff2, Ff3}
                         │
                         ▼
         Multi-scale Gaussian Anatomy Tokenizer (AGAT)
              ├── coarse tokens from F3
              └── middle tokens from F2
                         │
                         ▼
       Gaussian 2-Wasserstein + Log-Sinkhorn Matching
              ├── transport matrices P3, P2
              ├── token target barycenters
              ├── token displacements d3, d2
              └── token confidence c3, c2
                         │
                         ▼
       Gaussian-to-Volume Propagation at each scale
              ├── Gaussian displacement prior U3, U2
              └── Gaussian confidence prior C3, C2
                         │
                         ▼
       U-Net Residual Stationary Velocity Decoder
                         │
                         ▼
           Scaling-and-Squaring Integration
              ├── phi_fwd = exp(v)
              └── phi_inv = exp(-v)
                         │
                         ▼
       Spatial Transformer + Warped Moving Volume
```

模型由七个部分组成：

1. Shared multi-scale 3D encoder；
2. Multi-scale Gaussian Anatomy Tokenizer；
3. Gaussian 2-Wasserstein + Sinkhorn matching；
4. Gaussian-to-volume displacement propagation；
5. U-Net residual velocity decoder；
6. Scaling-and-squaring diffeomorphic integration；
7. Registration and Gaussian-specific training objectives。

---

# 3. 输入、输出与预处理约定

## 3.1 输入

```python
moving: Tensor[B, 1, D, H, W]
fixed: Tensor[B, 1, D, H, W]
moving_seg: Optional[Tensor[B, K, D, H, W]]
fixed_seg: Optional[Tensor[B, K, D, H, W]]
spacing: Optional[Tensor[B, 3]]  # order: z, y, x; first version should be identical after resampling
```

## 3.2 推荐预处理

第一版强制执行：

1. 按 DICOM 方向统一 orientation；
2. 重采样到统一 spacing；建议从 `1.5–2.0 mm` 各向同性开始；
3. 使用身体区域或头颈 ROI 做固定大小 crop/pad；
4. CT 强度 clip 后归一化到 `[-1, 1]`；
5. CBCT 可采用相同 clip，或使用 robust percentile normalization；
6. 在进入模型前完成 rigid/affine pre-alignment；
7. moving/fixed 的最终 shape 必须一致。

推荐首个实验体积尺寸：

```yaml
volume_size: [160, 192, 160]  # D, H, W，可按显存修改
batch_size: 1
```

## 3.3 模型输出

```python
{
    "warped_moving": Tensor[B, 1, D, H, W],
    "velocity": Tensor[B, 3, D, H, W],
    "phi_fwd": Tensor[B, D, H, W, 3],
    "phi_inv": Tensor[B, D, H, W, 3],
    "tokens_moving": Dict[str, GaussianTokenBatch],
    "tokens_fixed": Dict[str, GaussianTokenBatch],
    "matches": Dict[str, GaussianMatchOutput],
    "gaussian_priors": Dict[str, Tensor],
    "debug": Optional[Dict[str, Tensor]],
}
```

---

# 4. 模块一：Shared Multi-scale 3D Encoder

## 4.1 默认实现

第一版使用共享权重的 3D U-Net encoder；Swin backend 只保留接口，不作为首个里程碑。

推荐通道：

| Level | Resolution | Channels | Name |
|---|---:|---:|---|
| 0 | full | 16 | `F0` |
| 1 | 1/2 | 32 | `F1` |
| 2 | 1/4 | 64 | `F2` |
| 3 | 1/8 | 128 | `F3` |

每个 level：

```text
Conv3d(3x3x3) -> InstanceNorm3d -> LeakyReLU
Conv3d(3x3x3) -> InstanceNorm3d -> LeakyReLU
```

下采样使用：

```python
Conv3d(kernel_size=3, stride=2, padding=1)
```

类接口：

```python
class SharedRegistrationEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [F0, F1, F2, F3]."""
```

使用方式：

```python
Fm = encoder(moving)
Ff = encoder(fixed)
```

## 4.2 CT–CBCT 可选增强

若 CT–CBCT 域差明显，可替换第一层为 modality-specific stem：

```text
CT stem ──┐
          ├── shared encoder stages 1–3
CBCT stem ┘
```

第一版默认共享 encoder；只有在基线确认域差显著后再启用双 stem。

---

# 5. 模块二：Multi-scale Gaussian Anatomy Tokenizer（AGAT）

## 5.1 作用

将稠密 feature map 转换为固定数量、可微、具有方向和尺度的各向异性 Gaussian anatomy tokens。

每个 token：

\[
\mathcal G_i=(\mu_i,\Sigma_i,f_i,a_i).
\]

其中：

- `mu`：Gaussian center；
- `Sigma`：3×3 SPD covariance；
- `feat`：L2-normalized token feature；
- `anat_logits`：可选 anatomy class logits。

## 5.2 使用尺度

```yaml
token_scales:
  coarse:
    feature_level: 3
    token_grid: [4, 4, 4]   # N=64
  middle:
    feature_level: 2
    token_grid: [6, 6, 6]   # N=216
```

不要在第一版加入 full-resolution Gaussian tokens；配对矩阵和传播会导致显存过高。

## 5.3 数据结构

```python
@dataclass
class GaussianTokenBatch:
    mu: torch.Tensor          # [B, N, 3], normalized xyz in [-1, 1]
    sigma: torch.Tensor       # [B, N, 3], positive principal-axis std
    rotation: torch.Tensor    # [B, N, 3, 3]
    cov: torch.Tensor         # [B, N, 3, 3]
    feat: torch.Tensor        # [B, N, Ct], L2 normalized
    anat_logits: torch.Tensor # [B, N, Ka], optional; Ka may be 0
    offset: torch.Tensor      # [B, N, 3], for diagnostics
```

## 5.4 Anchor-grid 初始化

为了稳定训练，不使用不可控的 global top-K。每个尺度使用规则 token grid，并允许 token 在所属 cell 内学习偏移。

设 token grid 为 `(Gd, Gh, Gw)`，为每个 cell 生成中心 anchor：

```python
anchor_mu: [1, N, 3]  # xyz, range [-1, 1]
cell_size: [1, 1, 3]  # normalized xyz extent of one cell
```

预测偏移：

\[
\Delta\mu_i=0.35\,s_{cell}\odot\tanh(\hat\Delta\mu_i),
\]

\[
\mu_i=\mu_i^{anchor}+\Delta\mu_i.
\]

偏移限制在 cell 内，避免多个 token 全部坍缩到同一高对比区域。

## 5.5 协方差参数化

预测三个主轴尺度和 6D rotation：

\[
\Sigma_i=R_i\operatorname{diag}(\sigma_{i,1}^2,\sigma_{i,2}^2,\sigma_{i,3}^2)R_i^\top+\epsilon I.
\]

尺度定义：

\[
\sigma_i=s_{cell}\odot
\left(
\sigma_{min}+(
\sigma_{max}-\sigma_{min})\operatorname{sigmoid}(\hat\sigma_i)
\right).
\]

默认：

```yaml
sigma_min_ratio: 0.20
sigma_max_ratio: 1.20
cov_eps: 1.0e-5
```

旋转使用 6D continuous rotation representation，并通过 Gram–Schmidt 转为 `SO(3)`。

禁止使用 Euler angle。

## 5.6 Token feature 生成

为了让 Gaussian 方向真正影响 token feature，采用轻量 Gaussian-axis sampling，而不是只使用 adaptive average pooling。

流程：

1. 将 feature 投影到 `token_dim=96`；
2. adaptive average pooling 到 token grid，得到 base token feature；
3. 从 base token feature 预测 `mu/sigma/rotation`；
4. 沿 Gaussian 三个主轴采样 7 个点：

```text
0,
+e1, -e1,
+e2, -e2,
+e3, -e3
```

采样位置：

\[
p_{i,m}=\mu_i+\alpha R_i\operatorname{diag}(\sigma_i)\xi_m,
\qquad \alpha=0.75.
\]

5. 使用 `grid_sample` 从投影后的 feature map 采样；
6. 使用固定 Gaussian 权重加权；
7. 与 base feature 残差融合、LayerNorm、L2 normalize。

接口：

```python
class GaussianAnatomyTokenizer(nn.Module):
    def forward(self, feature: torch.Tensor) -> GaussianTokenBatch:
        ...
```

## 5.7 Anatomy logits

若训练数据有分割标签，可设置：

```yaml
num_anatomy_classes: K
use_anatomy_head: true
```

Anatomy logits 可表示器官类别，也可表示较粗的结构类型：

```text
bone / tubular / sheet-surface / soft-tissue / other
```

若无标签：

```yaml
use_anatomy_head: false
num_anatomy_classes: 0
```

第一版不能依赖 segmentation 作为推理输入；segmentation 只能作为训练辅助监督。

---

# 6. 模块三：Gaussian 2-Wasserstein + Sinkhorn Matching

## 6.1 匹配输入

对每个尺度独立匹配：

```python
moving_tokens: GaussianTokenBatch
fixed_tokens: GaussianTokenBatch
```

## 6.2 Gaussian 2-Wasserstein 距离

两个 Gaussian：

\[
\mathcal N(\mu_i,\Sigma_i),
\qquad
\mathcal N(\mu_j,\Sigma_j)
\]

其平方 2-Wasserstein 距离：

\[
W_2^2(i,j)
=
\|\mu_i-\mu_j\|_2^2
+
\operatorname{Tr}
\left(
\Sigma_i+\Sigma_j
-2(\Sigma_i^{1/2}\Sigma_j\Sigma_i^{1/2})^{1/2}
\right).
\]

实现要求：

```python
@torch.cuda.amp.autocast(enabled=False)
def pairwise_gaussian_w2(...):
    ...
```

SPD square root：

```python
def sqrtm_spd(matrix, eps=1e-6):
    eigvals, eigvecs = torch.linalg.eigh(matrix.float())
    eigvals = eigvals.clamp_min(eps)
    return eigvecs @ torch.diag_embed(eigvals.sqrt()) @ eigvecs.transpose(-1, -2)
```

输出：

```python
center_cost: [B, Nm, Nf]
cov_cost: [B, Nm, Nf]
w2_cost: [B, Nm, Nf]
```

## 6.3 混合匹配代价

\[
C_{ij}
=
\lambda_\mu\tilde C^\mu_{ij}
+
\lambda_\Sigma\tilde C^\Sigma_{ij}
+
\lambda_f C^f_{ij}
+
\lambda_a C^a_{ij}.
\]

其中：

\[
C^f_{ij}=1-\cos(f_i^m,f_j^f).
\]

若启用 anatomy head：

\[
C^a_{ij}=1-\langle p_i^m,p_j^f\rangle.
\]

为避免不同 cost 数量级不一致，中心和 covariance cost 需按 batch、scale 做 detached mean normalization：

```python
cost_norm = cost / (cost.detach().mean(dim=(-2, -1), keepdim=True) + 1e-6)
```

默认权重：

```yaml
matching_cost:
  lambda_center: 1.0
  lambda_covariance: 0.5
  lambda_feature: 1.0
  lambda_anatomy: 0.2
```

## 6.4 Log-domain Sinkhorn

第一版使用 balanced entropic optimal transport，moving/fixed token 数量可不同。

均匀边缘分布：

\[
r_i=1/N_m,
\qquad
c_j=1/N_f.
\]

求解：

\[
P^*=\arg\min_{P\in\Pi(r,c)}
\langle P,C\rangle-\varepsilon H(P).
\]

默认：

```yaml
sinkhorn:
  epsilon: 0.07
  iterations: 30
  convergence_tol: 1.0e-4
```

接口：

```python
class LogSinkhornMatcher(nn.Module):
    def forward(
        self,
        moving_tokens: GaussianTokenBatch,
        fixed_tokens: GaussianTokenBatch,
    ) -> "GaussianMatchOutput":
        ...
```

## 6.5 匹配输出

```python
@dataclass
class GaussianMatchOutput:
    transport: torch.Tensor       # [B, Nm, Nf]
    row_prob: torch.Tensor        # [B, Nm, Nf], row-normalized
    target_mu: torch.Tensor       # [B, Nm, 3]
    displacement: torch.Tensor    # [B, Nm, 3]
    confidence: torch.Tensor      # [B, Nm, 1]
    cost: torch.Tensor            # [B, Nm, Nf]
```

目标 barycenter：

\[
Q_{ij}=\frac{P_{ij}}{\sum_jP_{ij}+\epsilon},
\]

\[
\hat\mu_i^f=\sum_jQ_{ij}\mu_j^f,
\qquad
d_i=\hat\mu_i^f-\mu_i^m.
\]

置信度使用归一化 row entropy：

\[
c_i=1-
\frac{-\sum_jQ_{ij}\log(Q_{ij}+\epsilon)}
{\log N_f}.
\]

并 clamp 到 `[0, 1]`。

## 6.6 中尺度空间门控

为减少错误的远距离中尺度匹配，可对 middle scale 增加软空间门控：

```python
spatial_mask = pairwise_center_distance < spatial_radius
cost = cost + (~spatial_mask) * large_cost
```

默认仅 middle scale 开启：

```yaml
middle_spatial_radius: 1.0  # normalized coordinate distance，需验证后调整
large_cost: 1.0e4
```

coarse scale 不使用强门控，以保留大位移搜索能力。

---

# 7. 模块四：Gaussian-to-Volume Propagation（GVP）

## 7.1 作用

将稀疏 token displacement 转换成每个 token scale 上的稠密 Gaussian displacement prior 和 confidence prior。

对网格位置 `x`：

\[
w_i(x)
=
c_i
\exp
\left[
-\frac12
(x-\mu_i)^\top\Sigma_i^{-1}(x-\mu_i)
\right].
\]

位移先验：

\[
U_G(x)
=
\frac{\sum_iw_i(x)d_i}
{\sum_iw_i(x)+\epsilon}.
\]

置信度先验：

\[
C_G(x)
=
\frac{\sum_iw_i(x)}
{1+\sum_iw_i(x)}.
\]

## 7.2 输出尺度

- coarse prior：与 `F3` 相同空间尺寸；
- middle prior：与 `F2` 相同空间尺寸。

输出：

```python
U3: [B, 3, D3, H3, W3]
C3: [B, 1, D3, H3, W3]
U2: [B, 3, D2, H2, W2]
C2: [B, 1, D2, H2, W2]
```

位移通道顺序统一为 `(x, y, z)`。

## 7.3 显存要求

禁止一次创建完整 `[B, N, D, H, W]` 权重张量。

必须采用 token chunk：

```yaml
propagation_token_chunk: 32
```

伪代码：

```python
num = zeros(B, 3, D, H, W)
den = zeros(B, 1, D, H, W)
for token_chunk in chunks(tokens, chunk_size):
    mahal = ...
    weights = confidence * exp(-0.5 * mahal)
    num += sum(weights * displacement)
    den += sum(weights)
U = num / (den + eps)
C = den / (1.0 + den)
```

将 Mahalanobis distance clamp 到合理范围，例如 `[0, 30]`，避免 `exp` 下溢和梯度异常。

接口：

```python
class GaussianToVolumePropagator(nn.Module):
    def forward(
        self,
        tokens: GaussianTokenBatch,
        match: GaussianMatchOutput,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

---

# 8. 模块五：U-Net Residual Stationary Velocity Decoder

## 8.1 作用

Gaussian priors 只提供稀疏对应引导；最终 velocity 必须由稠密 decoder 结合 moving/fixed features 预测。

Gaussian displacement prior 只作为条件输入，不直接当作最终 velocity，避免将有限位移与 stationary velocity 混为一谈。

## 8.2 Decoder 输入

在 level 3：

```text
Fm3
Ff3
abs(Fm3 - Ff3)
U3
C3
```

在 level 2：

```text
upsampled decoder feature
Fm2
Ff2
abs(Fm2 - Ff2)
U2
C2
```

在 level 1 和 0：

```text
upsampled decoder feature
FmL
FfL
abs(FmL - FfL)
```

## 8.3 推荐结构

```text
Level 3 fusion -> ConvBlock(256)
Upsample x2
Level 2 fusion -> ConvBlock(128)
Upsample x2
Level 1 fusion -> ConvBlock(64)
Upsample x2
Level 0 fusion -> ConvBlock(32)
VelocityHead: Conv3d(32, 3, kernel_size=3, padding=1)
```

Upsample 可使用：

```python
trilinear interpolate + Conv3d
```

优先于 transposed convolution，以减少 checkerboard artifact。

Velocity head 初始化：

```python
nn.init.zeros_(velocity_head.weight)
nn.init.zeros_(velocity_head.bias)
```

输出：

```python
velocity: [B, 3, D, H, W]  # normalized xyz displacement per unit time
```

## 8.4 可选轻量 residual prior injection

默认不直接执行 `velocity += U_G`。

若后续消融发现 Gaussian prior 利用不足，可增加：

```python
velocity = velocity_head(decoder_feature) + beta * prior_projection(U2_up)
```

其中 `beta` 为可学习标量，初始化为 0。第一版默认关闭。

---

# 9. 模块六：Scaling-and-Squaring Diffeomorphic Integration

## 9.1 Identity grid

```python
identity_grid(shape) -> [B, D, H, W, 3]
```

坐标顺序 `(x, y, z)`，范围 `[-1, 1]`。

## 9.2 指数映射

对 velocity：

```python
phi = identity + velocity.permute(0, 2, 3, 4, 1) / (2 ** n_steps)
for _ in range(n_steps):
    phi = compose(phi, phi)
```

默认：

```yaml
integration_steps: 7
```

其中：

```python
compose(phi_a, phi_b) = sample(phi_a, phi_b)
```

语义：先应用 `phi_b`，再应用 `phi_a`。

计算：

```python
phi_fwd = integrate_velocity(+velocity)
phi_inv = integrate_velocity(-velocity)
```

## 9.3 Spatial transformer

```python
warped_moving = F.grid_sample(
    moving,
    phi_inv,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
```

分割标签使用 one-hot + trilinear，以保持可微；仅推理可使用 nearest。

接口：

```python
class DiffeomorphicIntegrator(nn.Module):
    def forward(self, velocity):
        return phi_fwd, phi_inv
```

---

# 10. 完整模型接口

```python
class GAMReg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SharedRegistrationEncoder(...)
        self.tokenizer_coarse = GaussianAnatomyTokenizer(...)
        self.tokenizer_middle = GaussianAnatomyTokenizer(...)
        self.matcher_coarse = LogSinkhornMatcher(...)
        self.matcher_middle = LogSinkhornMatcher(...)
        self.propagator = GaussianToVolumePropagator(...)
        self.decoder = ResidualVelocityDecoder(...)
        self.integrator = DiffeomorphicIntegrator(...)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_seg: torch.Tensor | None = None,
        fixed_seg: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        ...
```

## 10.1 Forward 伪代码

```python
Fm = encoder(moving)  # [Fm0, Fm1, Fm2, Fm3]
Ff = encoder(fixed)

Tm3 = tokenizer_coarse(Fm[3])
Tf3 = tokenizer_coarse(Ff[3])
Tm2 = tokenizer_middle(Fm[2])
Tf2 = tokenizer_middle(Ff[2])

M3 = matcher_coarse(Tm3, Tf3)
M2 = matcher_middle(Tm2, Tf2)

U3, C3 = propagator(Tm3, M3, Fm[3].shape[-3:])
U2, C2 = propagator(Tm2, M2, Fm[2].shape[-3:])

velocity = decoder(Fm, Ff, U3, C3, U2, C2)
phi_fwd, phi_inv = integrator(velocity)
warped_moving = spatial_transform(moving, phi_inv)

return {
    "warped_moving": warped_moving,
    "velocity": velocity,
    "phi_fwd": phi_fwd,
    "phi_inv": phi_inv,
    "tokens_moving": {"coarse": Tm3, "middle": Tm2},
    "tokens_fixed": {"coarse": Tf3, "middle": Tf2},
    "matches": {"coarse": M3, "middle": M2},
    "gaussian_priors": {"U3": U3, "C3": C3, "U2": U2, "C2": C2},
}
```

---

# 11. 损失函数

总损失：

\[
\mathcal L
=
\lambda_{sim}\mathcal L_{LNCC}
+
\lambda_{feat}\mathcal L_{feat}
+
\lambda_{smooth}\mathcal L_{smooth}
+
\lambda_{jac}\mathcal L_{jac}
+
\lambda_{dice}\mathcal L_{dice}
+
\lambda_{anchor}\mathcal L_{anchor}
+
\lambda_{anat}\mathcal L_{anat}
+
\lambda_{token}\mathcal L_{token-reg}.
\]

其中前五项是配准主损失；后三项用于保证 Gaussian 模块可训练且真正参与最终形变。

## 11.1 LNCC similarity

\[
\mathcal L_{LNCC}
=-\operatorname{LNCC}(I_f,I_{m\rightarrow f}).
\]

默认 window：

```yaml
lncc_window: [9, 9, 9]
```

CT–CBCT 若原始强度不稳定，可将 LNCC 与 feature similarity 同时使用，而不是完全删除 LNCC。

## 11.2 Multi-scale feature similarity

先使用最终 `phi_inv` 将 moving feature warp 到 fixed frame，再计算 level 2 和 3 的 cosine loss：

\[
\mathcal L_{feat}
=
\sum_{l\in\{2,3\}}
\left(
1-
\cos(
F_f^l,
F_m^l\circ\phi_{inv}^l
)
\right).
\]

注意：需要将 absolute inverse grid resize 到对应 feature resolution；位移使用归一化坐标时不需要额外按倍率缩放。

## 11.3 Smoothness

对 stationary velocity 做 spacing-aware first-order diffusion：

\[
\mathcal L_{smooth}
=
\sum_x
\|\nabla v(x)\|_2^2.
\]

第一版由于已重采样为各向同性，可按 voxel finite difference 实现。

## 11.4 Jacobian folding penalty

对 forward transform：

\[
\mathcal L_{jac}
=
\frac1{|\Omega|}
\sum_x
\operatorname{ReLU}
\left(
-\det J_{\phi_{fwd}}(x)
\right)^2.
\]

计算 Jacobian 前，必须将 normalized coordinate transform 转换为 voxel coordinate transform，否则不同图像尺寸的数值不可比。

同时记录：

```python
folding_ratio = mean(detJ <= 0)
```

## 11.5 Dice loss

仅在 moving/fixed segmentation 可用时启用：

\[
\mathcal L_{dice}
=1-\operatorname{Dice}
(S_f,S_m\circ\phi_{inv}).
\]

- 使用 one-hot segmentation；
- 排除 background；
- warp 时使用 trilinear；
- 训练和验证分别报告 per-organ Dice。

## 11.6 Gaussian anchor consistency

这是第一版必须加入的 Gaussian-specific loss。

在 moving token center 采样 forward transform：

\[
\tilde\mu_i^f
=
\phi_{fwd}(\mu_i^m).
\]

约束：

\[
\mathcal L_{anchor}
=
\frac{
\sum_i c_i
\|\tilde\mu_i^f-\hat\mu_i^f\|_1
}{
\sum_i c_i+\epsilon
}.
\]

coarse 和 middle 两个尺度分别计算后相加。

该项确保：

```text
Gaussian matching -> final dense deformation
```

之间存在直接监督链，避免 decoder 完全忽略 Gaussian priors。

## 11.7 Anatomy token loss

若有 anatomy labels：

1. 将 segmentation 在 token Gaussian 位置做加权采样；
2. 得到 token-level class distribution；
3. 使用 cross entropy 或 KL loss：

\[
\mathcal L_{anat}
=
\operatorname{CE}(a_i,y_i^{token}).
\]

无标签时该项为 0。

## 11.8 Token regularization

由于使用 anchor-grid，第一版只需两个稳定项：

偏移正则：

\[
\mathcal L_{offset}=\frac1N\sum_i\|\Delta\mu_i\|_2^2.
\]

尺度条件数正则：

\[
\mathcal L_{cond}
=
\frac1N\sum_i
\operatorname{ReLU}
\left(
\frac{\max_k\sigma_{ik}}
{\min_k\sigma_{ik}+\epsilon}
-\kappa
\right)^2,
\]

默认 `kappa=8`。

\[
\mathcal L_{token-reg}
=\mathcal L_{offset}+0.1\mathcal L_{cond}.
\]

## 11.9 默认权重

```yaml
loss_weights:
  sim: 1.0
  feature: 0.20
  smooth: 0.05
  jacobian: 0.10
  dice: 1.0
  anchor: 0.50
  anatomy: 0.10
  token_regularization: 0.01
```

这些是起始值，不应视为最终最优值。推荐搜索范围：

```text
smooth: 0.01–0.20
jacobian: 0.01–0.20
anchor: 0.10–1.00
feature: 0.10–0.50
```

---

# 12. 训练策略

## 12.1 推荐三阶段训练

### Stage A：Synthetic token warm-up

目标：先让 Gaussian tokenizer 和 matcher 学会几何对应。

方法：

1. 从同一 volume 构造 source/target；
2. 对 source 施加已知 random affine + smooth elastic transform；
3. 获得 token 真实目标位置或真实 dense transform；
4. 训练 token center、covariance、feature 和 matching；
5. 使用：`anchor loss + anatomy loss + token regularization + feature loss`。

建议：

```yaml
warmup_iterations: 10000-20000
```

### Stage B：Registration warm-up

目标：稳定 dense decoder 和 diffeomorphic integration。

做法：

- 使用真实 moving/fixed pairs；
- Gaussian priors正常输入；
- `anchor weight` 从 0.1 线性增加到 0.5；
- `jacobian weight` 从 0 线性增加到目标值；
- velocity head 保持零初始化。

### Stage C：Joint fine-tuning

全部模块联合训练：

```text
encoder + tokenizer + matcher + propagator + decoder + integrator
```

若 Sinkhorn 在早期不稳定，可前 2000 iterations 暂时只使用 feature cost 和 center cost，随后逐渐加入 covariance cost。

## 12.2 优化器

```yaml
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 1.0e-5
gradient_clip_norm: 1.0
scheduler: cosine
mixed_precision: true
```

例外：

- Gaussian SPD sqrt；
- Jacobian determinant；
- Sinkhorn log normalization 的关键步骤；

建议在 float32 中执行。

## 12.3 数据增强

推荐：

- small random rotation；
- translation；
- isotropic/anisotropic scaling；
- smooth elastic deformation；
- gamma/intensity shift；
- Gaussian noise；
- CBCT-like streak/noise augmentation；
- random crop，但不能破坏 fixed/moving 对应关系。

强度增强可以分别施加；几何增强必须保存或同步更新 ground-truth transform/segmentation。

---

# 13. 推荐配置文件

Codex 应创建配置解析，并支持以下默认配置：

```yaml
model:
  name: GAMReg
  in_channels: 1
  encoder_channels: [16, 32, 64, 128]
  token_dim: 96
  use_anatomy_head: true
  num_anatomy_classes: 5

  tokenizers:
    coarse:
      feature_level: 3
      token_grid: [4, 4, 4]
      sigma_min_ratio: 0.20
      sigma_max_ratio: 1.20
      offset_ratio: 0.35
    middle:
      feature_level: 2
      token_grid: [6, 6, 6]
      sigma_min_ratio: 0.20
      sigma_max_ratio: 1.20
      offset_ratio: 0.35

  matching:
    lambda_center: 1.0
    lambda_covariance: 0.5
    lambda_feature: 1.0
    lambda_anatomy: 0.2
    sinkhorn_epsilon: 0.07
    sinkhorn_iterations: 30
    middle_spatial_radius: 1.0

  propagation:
    token_chunk: 32
    mahalanobis_clip: 30.0

  integration:
    steps: 7

loss:
  lncc_window: [9, 9, 9]
  weights:
    sim: 1.0
    feature: 0.20
    smooth: 0.05
    jacobian: 0.10
    dice: 1.0
    anchor: 0.50
    anatomy: 0.10
    token_regularization: 0.01

training:
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.00001
  gradient_clip_norm: 1.0
  amp: true
  batch_size: 1
```

---

# 14. 推荐代码目录

```text
gam_reg/
├── configs/
│   └── gam_reg_stable.yaml
├── models/
│   ├── encoder.py
│   ├── gaussian_types.py
│   ├── gaussian_tokenizer.py
│   ├── gaussian_wasserstein.py
│   ├── sinkhorn.py
│   ├── gaussian_matcher.py
│   ├── gaussian_propagation.py
│   ├── velocity_decoder.py
│   ├── diffeomorphic.py
│   ├── spatial_transformer.py
│   └── gam_reg.py
├── losses/
│   ├── lncc.py
│   ├── feature_similarity.py
│   ├── deformation_losses.py
│   ├── dice.py
│   ├── gaussian_losses.py
│   └── total_loss.py
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── augmentations.py
├── metrics/
│   ├── registration_metrics.py
│   ├── jacobian_metrics.py
│   └── landmark_metrics.py
├── tests/
│   ├── test_coordinate_convention.py
│   ├── test_gaussian_w2.py
│   ├── test_sinkhorn.py
│   ├── test_propagation.py
│   ├── test_diffeomorphic.py
│   ├── test_gradients.py
│   └── test_model_smoke.py
├── train.py
├── validate.py
└── infer.py
```

---

# 15. 必须实现的单元测试

## 15.1 Coordinate convention test

构造一个点向 `+x` 平移的已知 grid，验证：

- tensor 维度仍为 `[D,H,W]`；
- grid 最后一维 `[0]` 的确控制 x；
- moving warp 到 fixed 的方向正确；
- forward map 与 inverse map 没有颠倒。

## 15.2 Gaussian W2 test

必须满足：

```text
W2(G, G) ≈ 0
W2(G1, G2) = W2(G2, G1)
W2 >= 0
```

并测试对 `mu`、`sigma` 和 `rotation` 均可反向传播。

## 15.3 Sinkhorn test

对随机小矩阵：

```text
P >= 0
row sums ≈ uniform moving marginal
column sums ≈ uniform fixed marginal
no NaN / inf
```

## 15.4 Propagation test

单 token、固定 covariance 和 displacement 情况下：

- token center 处输出接近该 displacement；
- 远离 token 后 confidence 下降；
- 沿长主轴的影响范围大于短主轴。

## 15.5 Diffeomorphic integration test

- `v=0` 时 `phi_fwd=phi_inv=identity`；
- 小常量平移时结果接近期望平移；
- `compose(phi_fwd, phi_inv)` 接近 identity；
- identity pair 的 Jacobian 接近 1。

## 15.6 End-to-end gradient test

对总 loss 反向传播，以下参数必须有非零有限梯度：

```text
token center head
token scale head
rotation head
token feature projection
encoder
velocity decoder
```

## 15.7 Smoke test

对小体积，例如 `[1,1,32,40,32]`：

- forward 成功；
- output shapes 正确；
- loss 可计算；
- backward 成功；
- 无 NaN/inf。

---

# 16. 训练与验证日志

每个 epoch 至少记录：

```text
train/val total loss
LNCC loss
feature loss
smoothness loss
Jacobian penalty
Dice loss
anchor loss
mean token confidence
mean/max Gaussian sigma
mean Gaussian anisotropy ratio
Sinkhorn entropy
folding ratio
mean absolute velocity
```

验证阶段至少报告：

- Dice per organ；
- mean Dice；
- landmark TRE（若有 landmarks）；
- HD95/ASSD；
- folding ratio；
- mean `|detJ - 1|`；
- inference time；
- GPU memory。

同时保存可视化：

1. fixed/moving/warped moving 的三正交切片；
2. deformation magnitude；
3. Jacobian determinant；
4. Gaussian centers 和主轴投影；
5. token confidence；
6. coarse/middle Gaussian displacement priors。

---

# 17. SCI 论文需要的核心消融

必须至少实现以下模型变体：

| Variant | 目的 |
|---|---|
| Baseline U-Net registration | 无 Gaussian 的基础模型 |
| Point Tokens | 去掉 covariance，仅使用 center + feature |
| Isotropic Gaussian | `sigma1=sigma2=sigma3` |
| Anisotropic Gaussian without W2 | covariance 只用于传播，匹配不用 W2 |
| Anisotropic Gaussian + W2, no Sinkhorn | 使用 softmax/nearest matching |
| Full GAM-Reg | AGAT + W2 + Sinkhorn + GVP + decoder |
| Full without anchor loss | 验证 Gaussian 是否会被 decoder 忽略 |
| Full without Dice | 验证非标签监督能力 |

最关键的因果链：

```text
Point token
    -> Isotropic Gaussian
    -> Anisotropic Gaussian
    -> + Gaussian W2
    -> + Sinkhorn
    -> + Gaussian-to-volume propagation
    -> + anchor consistency
```

每一步都应在 Dice/TRE/HD95/folding ratio 上给出定量结果。

---

# 18. 首版性能与稳定性验收标准

以下不是最终论文目标，而是工程是否合格的最低标准：

1. identity image pair 上平均位移接近 0；
2. synthetic affine/elastic pair 上显著优于无配准输入；
3. full model 不低于基础 U-Net registration；
4. Gaussian 模块参数获得稳定梯度；
5. Sinkhorn 不出现 NaN；
6. folding ratio 不高于 baseline；
7. 开启 anchor loss 后 token displacement 与 final transform 的一致性明显提高；
8. anisotropic propagation 的可视化应与主轴方向一致；
9. inference 不依赖 segmentation；
10. batch size 1 下可在单 GPU 完成训练与推理。

---

# 19. 第一版禁止事项

为保证论文主线和工程可控，第一版禁止加入：

- diffusion refinement；
- reinforcement learning；
- language/VLM modules；
- Gaussian rendering 或 3DGS rasterizer；
- longitudinal sequence model；
- source–sink/metamorphosis；
- learned uncertainty posterior；
- 过多 organ-specific expert heads；
- full-resolution all-to-all Gaussian matching。

这些内容应在 GAM-Reg 稳定、消融完成后再考虑。

---

# 20. 论文中的模块定位

Gaussian 模块在本模型中不是最终 deformation predictor，而是：

> **介于稠密编码器与形变解码器之间的结构化对应前端。**

它承担四项连续职责：

1. 用各向异性 Gaussian 显式表示局部解剖位置、尺度和方向；
2. 用 Gaussian 2-Wasserstein 与 feature cost 建立结构感知对应；
3. 用 Sinkhorn 得到软匹配及匹配置信度；
4. 用 Gaussian covariance 将稀疏位移按解剖方向传播到体素空间，并约束最终稠密形变。

建议论文中的核心表述：

> Rather than treating tokens as geometry-free feature vectors, GAM-Reg represents each local anatomical element as an anisotropic Gaussian and uses the same geometry consistently for correspondence estimation, displacement propagation, and dense deformation conditioning.

---

# 21. Definition of Done

Codex 完成实现时，必须交付：

1. 按第 14 节目录组织的可运行工程；
2. `train.py`、`validate.py`、`infer.py`；
3. 一个默认 YAML 配置；
4. 所有第 15 节单元测试；
5. 一个小体积 synthetic demo；
6. checkpoint 保存与恢复；
7. TensorBoard 或等价日志；
8. 每个关键张量的 shape assertion；
9. README 中说明 moving/fixed 与 forward/inverse 方向；
10. 禁止用 placeholder、伪实现或恒等输出绕过 Gaussian W2、Sinkhorn、GVP 或 scaling-and-squaring。

---

# 22. 建议 Codex 的实现顺序

严格按以下顺序开发和测试：

1. `identity_grid`、`spatial_transformer`、坐标方向测试；
2. scaling-and-squaring 与 inverse consistency 测试；
3. shared 3D encoder；
4. anchor-grid Gaussian tokenizer；
5. SPD matrix sqrt 与 Gaussian W2；
6. log-domain Sinkhorn；
7. Gaussian match output；
8. chunked Gaussian-to-volume propagation；
9. residual velocity decoder；
10. end-to-end GAM-Reg forward；
11. LNCC/smoothness/Jacobian/Dice；
12. Gaussian anchor loss；
13. synthetic warm-up training；
14. real pair joint training；
15. ablation variants and evaluation scripts。

只有当前一步单元测试通过后，才进入下一步。
