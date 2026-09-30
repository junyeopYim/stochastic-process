# 표기 규약 (notation)

한 번 정하고 모든 노트북에서 그대로 씁니다. 바꾸고 싶으면 여기서 먼저 바꾸고, 바뀐 노트북 번호를 아래 변경 기록에 남깁니다.

## 이산시간 마르코프 사슬 (DTMC)
| 기호 | 의미 | 코드 |
|---|---|---|
| $X_n$, $n = 0, 1, 2, \dots$ | 시각 $n$의 상태 | `paths[:, n]` |
| $S = \{0, 1, \dots, N-1\}$ | 상태공간 (항상 0부터 시작하는 정수) | `n_states` |
| $P = (p_{ij})$, $p_{ij} = P(X_{n+1}=j \mid X_n=i)$ | 전이 행렬 (행 확률 1) | `P` (shape `(n, n)`) |
| $P^n$ | $n$단계 전이 행렬 | `np.linalg.matrix_power(P, n)` |
| $\pi$ | 정상분포 (행벡터, $\pi P = \pi$) | `pi` (shape `(n,)`) |
| $T_x = \min\{n \ge 1 : X_n = x\}$ | $x$로의 첫 (재)도달 시간 | `hitting_times(...)` |
| $\lambda_2$ | 두 번째로 큰 고유값의 절댓값 | `lam2` |
| $\|\mu - \nu\|_{TV}$ | 전변동 거리 $\tfrac12 \sum_x |\mu(x) - \nu(x)|$ | `tv_distance(mu, nu)` |

## 푸아송 과정 / 연속시간 마르코프 사슬 (CTMC)
| 기호 | 의미 | 코드 |
|---|---|---|
| $N(t)$ | $[0, t]$의 도착 횟수 | `renewal_count(times, t)` |
| $\lambda$, $\lambda(t)$, $\Lambda(t) = \int_0^t \lambda(s)\,ds$ | 강도(rate), 시간변동 강도, 누적 강도 | `rate`, `rate_fn`, `Lambda` |
| $\tau_i$ | $i$번째 도착 간격 (i.i.d. Exp($\lambda$)) | `gaps` |
| $Q = (q_{ij})$ | 생성원 (off-diagonal $\ge 0$, 행합 0) | `Q` |
| $q_i = -q_{ii}$ | 상태 $i$의 이탈률 (holding rate) | `holding` |
| $P(t) = e^{Qt}$ | 전이 확률 행렬 | `transition_matrix(Q, t)` |
| $\rho = \lambda / \mu$ (M/M/1), $\rho = \lambda/(c\mu)$ (M/M/c) | 이용률 | `rho` |
| $L, L_q, W, W_q$ | 시스템/대기열의 평균 고객 수, 평균 체류/대기 시간 | `mmc_metrics(...)` |

## 재생 과정 / 마팅게일
| 기호 | 의미 | 코드 |
|---|---|---|
| $\mu = E[\tau]$ | 평균 재생 간격 | `mu_tau` |
| $A(t), R(t)$ | 시각 $t$의 나이(age), 잔여수명(residual) | `age_residual_at(times, t)` |
| $\mathcal{F}_n$ | $n$까지의 정보 (filtration) | — |
| $M_n$ | 마팅게일 | `M` |
| $T$ | 정지 시간 (stopping time) | `T` |
| $E[X \mid \mathcal{F}_n]$ | 조건부 기대 (확률변수) | 구간화 평균으로 추정 |

## 브라운 운동 / SDE
| 기호 | 의미 | 코드 |
|---|---|---|
| $B_t$ (또는 $W_t$) | 표준 브라운 운동, $B_0 = 0$ | `B` (shape `(n_paths, n_steps+1)`) |
| $\Delta t = T / n$ | 시간 격자 간격 | `dt` |
| $\Delta B_k = B_{t_{k+1}} - B_{t_k} \sim N(0, \Delta t)$ | 증분 | `dW` |
| $[B]_t = t$ | 이차 변동 | `quadratic_variation(B)` |
| $dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dB_t$ | SDE (drift $b$, diffusion $\sigma$) | `drift(t, x)`, `diffusion(t, x)` |
| $S_t = S_0 \exp\big((\mu - \sigma^2/2) t + \sigma B_t\big)$ | 기하 브라운 운동 | `gbm_paths(...)` |

## 몬테카를로
| 기호 | 의미 | 코드 |
|---|---|---|
| $\hat\theta_N$ | $N$개 표본의 평균 | `MCResult.mean` |
| $\mathrm{SE} = s / \sqrt{N}$ | 표준오차 | `MCResult.se` |
| 95% CI | $\hat\theta \pm 1.96\,\mathrm{SE}$ | `MCResult.ci95` |
| 판정 | 표에서는 3 SE 밖이면 조사, `assert`는 4 SE | `compare_table`, `assert_close` |

## 시드 규칙
`SEED = 100 × 모듈번호 + 알파벳 순서` (a=1, …, r=18, x=24), 캡스톤은 `900 + 번호`.
예: `00a` → 1, `01d` → 104, `01r` → 118, `C1` → 901. 코드에서는 항상 `SEED, rng = rng_for("01d")`.

## 변경 기록
- (없음)
