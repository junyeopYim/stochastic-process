# stochastic-process

확률 과정(stochastic processes)을 **Jupyter 노트북 중심으로 스스로에게 가르치는** 약 6개월 과정입니다.
모든 노트북은 같은 골격(Recall → 이론 → **Predict** → Simulate → **Compare + assert** → 연습문제 → 정리)을 따르고,
시뮬레이션 결과는 항상 닫힌 형식(closed form)과 표준오차 안에서 대조합니다.

과정을 꿰는 질문은 하나입니다. **"이 과정은 오랜 시간 뒤에 어떻게 보이고, 그 사이 무엇이 보존되는가?"**
(정상분포 → 재생-보상 비율 → 선택적 정지 정리 → 이토 공식)

## 실행법

```bash
uv sync                      # Python 3.13 + 의존성 (uv가 없으면 https://docs.astral.sh/uv/ 참고)
uv run nbstripout --install  # 새 클론마다 한 번: 커밋 시 노트북 출력 제거 (git filter)
uv run jupyter lab           # notebooks/00_toolkit/00a_random_walk.ipynb 부터
```

검사:

```bash
uv run pytest                                   # spkit 단위 테스트 + 노트북 템플릿 검사
SPKIT_FAST=1 uv run pytest --nbmake notebooks/  # 모든 노트북 실행 (1/10 크기, 약 10~15분)
uv run ruff check src tests                     # 린트
```

`SPKIT_FAST=1`을 켜면 모든 노트북이 표본 수를 1/10로 줄여 빠르게 돕니다(CI가 이 모드로 매주 실행). 공부할 때는 끄고 전체 크기로 돌리세요.

## 저장소 구조

```
notebooks/     00_toolkit … 07_sde, capstones/   ← 과정 본체 (아래 진행표 순서)
src/spkit/     공용 시뮬레이션 헬퍼 (markov, poisson, ctmc, renewal, brownian, sde, mc, plots)
tests/         spkit 단위 테스트 + 노트북 템플릿 검사(test_notebooks_structure.py)
notes/         notation.md(표기 규약) · textbook_map.md(노트북→교재 절) · cheatsheet.md(모듈별 공식)
log/           mistakes.md(틀린 예측 기록) · progress.md(세션 기록)
```

## 공부하는 법 (노트북 하나 = 60~120분)

1. **Recall**: 지난 노트북 질문에 기억으로 답한 뒤 `<details>`를 엽니다.
2. **이론**: 책을 덮고 "✍️ 내 말로" 셀에 정의·정리를 다시 씁니다. "가정이 왜 필요한가"에 답합니다.
3. **Predict**: 코드를 돌리기 **전에** `predictions` 딕셔너리를 채웁니다. 틀리면 `log/mistakes.md`에 기록합니다. 놀람이 있어야 기억에 남습니다.
4. **Simulate → Compare**: 표에서 3 SE 밖이면 버그거나 가정 오해입니다. 둘 다 기록할 가치가 있습니다.
5. **연습문제 3개**: E1 계산, E2 유도 후 시뮬레이션 검증, E3 가정을 깨보기.
6. **정리** 후 `log/progress.md`에 한 줄, 커밋 메시지는 `nb(01d): stationary — done`.

주간 리듬: 세션 A(읽기+이론 60분) · B(시뮬레이션+비교 90분) · C(연습문제+정리 60분) · 일요일 15분(`grep -rn TODO notebooks/`).
한 주를 건너뛰었으면 새 내용이 아니라 복습 세션으로 재개합니다. 월 1회는 4주 전 노트북의 실험 부분을 가리고 공식 하나를 맹목 유도합니다.

## 규칙

- 시드: `SEED = 100 × 모듈번호 + 알파벳 순서`(캡스톤은 900+번호). 코드에서는 항상 `SEED, rng = rng_for("01d")`.
- 파일 번호는 재부여하지 않습니다(새 주제는 다음 번호 또는 `01c2`). 이 진행표가 정식 순서입니다.
- 산문은 한국어, 식별자·파일명·그림 텍스트는 영어(폰트 설정 없이 한글 깨짐을 피하는 규칙).
- 새 헬퍼는 같은 코드를 **세 번째** 복붙할 때 `src/spkit/`로 옮기고, 테스트 하나와 함께 옮깁니다.
- 노트북 출력은 커밋하지 않습니다(git filter). GitHub에서는 그림이 보이지 않으니 로컬에서 실행해 보세요.

## 교재 (판 고정)

Durrett *Essentials of Stochastic Processes* 3e(무료 PDF, 척추) · Ross *Introduction to Probability Models* 12e(직관·연습문제) · Lawler *Introduction to Stochastic Processes* 2e, Norris *Markov Chains*(증명) · Grimmett & Stirzaker *Probability and Random Processes* 4e(엄밀) · Øksendal *Stochastic Differential Equations* 6e, Shreve *Stochastic Calculus for Finance II*(SDE·금융).
주의: Durrett 6장은 브라운 운동이 아니라 *Mathematical Finance*입니다. 브라운 운동은 Lawler 8장 / Ross 10장 / G&S 13장으로 읽습니다.

Season 2(핵심 36개를 끝내기 전에는 손대지 않음): GP 회귀, Lévy 과정, HMM/Kalman, MDP, Girsanov, Fokker-Planck, 측도론적 재구성.

## 진행표

상태 열은 **내 학습 상태**입니다(☐ 미시작 → 진행 중 → ✅ 완료). 교재 열은 대표 절 두 개만 적었고 전체는 [notes/textbook_map.md](notes/textbook_map.md)에 있습니다. 노트북 수: 핵심 36 · 선택 3 · 복습 3 · 캡스톤 5 (총 47개; 선택 3개를 뺀 44개가 기본 과정).

| 노트북 | 종류 | 주제 | 교재 | 분 | 상태 |
|---|---|---|---|---|---|
| **00 Toolkit + 확률 복습** | | | | | |
| [00a](notebooks/00_toolkit/00a_random_walk.ipynb) | 핵심 | 랜덤 워크로 시작하기 | Durrett 3e 1.1 (Example 1.1 gambler's ruin) · Ross 12e 2.2.2 (binomial) | 90 | ☐ |
| [00b](notebooks/00_toolkit/00b_monte_carlo_error.ipynb) | 핵심 | 몬테카를로 오차와 신뢰구간 | Ross 12e 2.7 (limit theorems: Chebyshev · G&S 4e 5.10 (two limit theorems: LLN | 75 | ☐ |
| [00c](notebooks/00_toolkit/00c_conditioning_pgf_exponential.ipynb) | 핵심 | 조건부 기대, pgf, 지수분포 | Ross 12e 3.1-3.4 (conditional expectation · Durrett 3e 2.1 (exponential distribution: memoryless | 90 | ☐ |
| **01 이산시간 마르코프 사슬 (DTMC)** | | | | | |
| [01a](notebooks/01_dtmc/01a_definition_chapman_kolmogorov.ipynb) | 핵심 | 마르코프 사슬의 정의와 채프먼-콜모고로프 | Durrett 3e 1.1 Definitions and examples · Ross 12e 4.1 Introduction | 90 | ☐ |
| [01b](notebooks/01_dtmc/01b_classification.ipynb) | 핵심 | 상태의 분류 | Durrett 3e 1.3 Classification of states · Ross 12e 4.3 Classification of States | 90 | ☐ |
| [01c](notebooks/01_dtmc/01c_absorption_gamblers_ruin.ipynb) | 핵심 | 흡수와 도박꾼의 파산 | Durrett 3e 1.8 Exit distributions · Ross 12e 4.5.1 The Gambler's Ruin Problem | 90 | ☐ |
| [01d](notebooks/01_dtmc/01d_stationary.ipynb) | 핵심 | 정상분포 | Durrett 3e 1.4 Stationary distributions · Ross 12e 4.4 Long-Run Proportions and Limiting Probabilities | 90 | ☐ |
| [01e](notebooks/01_dtmc/01e_convergence_mixing.ipynb) | 핵심 | 수렴 정리와 혼합 | Durrett 3e 1.5 Limit behavior · Norris 1.8 Convergence to equilibrium | 100 | ☐ |
| [01f](notebooks/01_dtmc/01f_reversibility_mcmc.ipynb) | 핵심 | 가역성과 MCMC | Durrett 3e 1.6 Special examples (detailed balance · Ross 12e 4.8 Time Reversible Markov Chains | 100 | ☐ |
| [01g](notebooks/01_dtmc/01g_galton_watson.ipynb) | 핵심 | 골턴-왓슨 분기 과정 | Durrett 3e 1.10 Infinite state spaces (branching processes) · Lawler 2e 2.4 Branching process | 90 | ☐ |
| [01x](notebooks/01_dtmc/01x_srw_recurrence_by_dimension.ipynb) | 선택 | 차원별 단순 랜덤 워크의 재귀성 | Lawler 2e 2.2 Recurrence and transience (simple random walk in Z^d) · Durrett 3e 1.10 Infinite state spaces | 90 | ☐ |
| [01r](notebooks/01_dtmc/01r_review.ipynb) | 복습 | 모듈 01 복습 (닫힌 책) | Durrett 3e Ch. 1 (1.1–1.10) — 복습용 전체 · Ross 12e Ch. 4 (4.1–4.9) — 4장 끝 연습문제에서 추가 문제 선택 | 60 | ☐ |
| **캡스톤** | | | | | |
| [C1](notebooks/capstones/C1_pagerank.ipynb) | 캡스톤 | 캡스톤 1: PageRank | Durrett 3e 1.4 (stationary distributions) · Lawler 2e Ch. 1 (finite Markov chains: invariant probabilities | 120 | ☐ |
| **02 푸아송 과정** | | | | | |
| [02a](notebooks/02_poisson/02a_memoryless_competing_clocks.ipynb) | 핵심 | 무기억성과 경쟁하는 시계 | Ross 12e 5.2 (5.2.1 정의 · Durrett 3e 2.1 Exponential distribution | 75 | ☐ |
| [02b](notebooks/02_poisson/02b_three_definitions_order_statistics.ipynb) | 핵심 | 푸아송 과정의 세 정의와 순서통계량 | Durrett 3e 2.2 Defining the Poisson process · Ross 12e 5.3 (5.3.1 계수과정 | 90 | ☐ |
| [02c](notebooks/02_poisson/02c_thinning_superposition_nhpp.ipynb) | 핵심 | 세분화, 중첩, 비동질 푸아송 과정 | Durrett 3e 2.4 Transformations (thinning · Ross 12e 5.3.4 (Further properties: thinning Prop 5.2 | 90 | ☐ |
| [02d](notebooks/02_poisson/02d_compound_inspection_teaser.ipynb) | 핵심 | 복합 푸아송과 검사 역설 | Durrett 3e 2.3 Compound Poisson processes · Ross 12e 5.4.2 Compound Poisson process | 75 | ☐ |
| **03 연속시간 마르코프 사슬 + 대기행렬** | | | | | |
| [03a](notebooks/03_ctmc/03a_generator_gillespie.ipynb) | 핵심 | 생성원과 질레스피 알고리즘 | Durrett 3e 4.1 Definitions and examples · Ross 12e 6.2 Continuous-time Markov chains | 90 | ☐ |
| [03b](notebooks/03_ctmc/03b_kolmogorov_expm.ipynb) | 핵심 | 콜모고로프 방정식과 행렬 지수 | Durrett 3e 4.2 Computing the transition probability · Ross 12e 6.4 The transition probability function P_ij(t) | 90 | ☐ |
| [03c](notebooks/03_ctmc/03c_birth_death_stationary.ipynb) | 핵심 | 출생-사망 과정과 정상분포 | Durrett 3e 4.3 Limiting behavior · Ross 12e 6.3 Birth and death processes | 90 | ☐ |
| [03d](notebooks/03_ctmc/03d_mm1_little.ipynb) | 핵심 | M/M/1 대기행렬과 리틀의 법칙 | Durrett 3e 4.5 Markovian queues · Ross 12e 8.2 Preliminaries (cost equations | 90 | ☐ |
| [03e](notebooks/03_ctmc/03e_mmc_finite_capacity.ipynb) | 핵심 | M/M/c와 유한 용량 | Durrett 3e 4.5 Markovian queues (M/M/s · Ross 12e 8.3 Exponential models (finite capacity | 90 | ☐ |
| [03f](notebooks/03_ctmc/03f_uniformization.ipynb) | 선택 | 균일화 | Ross 12e 6.8 Uniformization · Norris 2.1 Q-matrices and their exponentials | 75 | ☐ |
| [03r](notebooks/03_ctmc/03r_review.ipynb) | 복습 | 모듈 02-03 복습 (닫힌 책) | Durrett 3e Ch. 2 (2.1–2.4) · Ross 12e 5.2–5.3 (exponential | 60 | ☐ |
| **캡스톤** | | | | | |
| [C2](notebooks/capstones/C2_queue_design.ipynb) | 캡스톤 | 캡스톤 2: 대기행렬 설계 | Durrett 3e 4.3 (limiting behavior of CTMC) · Ross 12e Ch. 8 (queueing theory): 8.2 (cost identities | 120 | ☐ |
| **04 재생 과정** | | | | | |
| [04a](notebooks/04_renewal/04a_elementary_renewal_reward.ipynb) | 핵심 | 기본 재생 정리와 재생-보상 | Durrett 3e 3.1 Laws of Large Numbers (N(t)/t → 1/μ · Ross 12e 7.1–7.4 (7.2 Distribution of N(t) | 90 | ☐ |
| [04b](notebooks/04_renewal/04b_age_residual_inspection_paradox.ipynb) | 핵심 | 나이, 잔여수명, 검사 역설 | Durrett 3e 3.3 Age and Residual Life (3.3.1 discrete case · Ross 12e 7.7 The Inspection Paradox | 90 | ☐ |
| [04c](notebooks/04_renewal/04c_regenerative_view_of_markov_chains.ipynb) | 핵심 | 재생 관점에서 본 마르코프 사슬 | Durrett 3e 1.5 Limit Behavior (asymptotic frequency N_n(y)/n → 1/E_yT_y · Ross 12e 4.4 Long-Run Proportions and Limiting Probabilities | 90 | ☐ |
| **05 마팅게일** | | | | | |
| [05a](notebooks/05_martingales/05a_conditional_expectation_examples_zoo.ipynb) | 핵심 | 조건부 기대와 마팅게일 동물원 | Durrett 3e 5.1 Conditional Expectation · Williams Ch.9 Conditional Expectation (9.7 properties list) | 90 | ☐ |
| [05b](notebooks/05_martingales/05b_optional_stopping_ruin_doubling.ipynb) | 핵심 | 선택적 정지 정리와 파산·배가 전략 | Durrett 3e 5.3 Gambling Strategies · Williams 10.6-10.7 (previsible processes | 100 | ☐ |
| [05c](notebooks/05_martingales/05c_wald_pattern_waiting.ipynb) | 핵심 | 월드 항등식과 패턴 대기시간 | Williams 10.10 Doob's Optional-Stopping Theorem · G&S 4e 10.2 (Wald's equation | 90 | ☐ |
| [05d](notebooks/05_martingales/05d_convergence_polya_urn.ipynb) | 핵심 | 마팅게일 수렴과 폴리아 항아리 | Durrett 3e 5.5 Convergence · Williams 11.1-11.5 (upcrossings | 90 | ☐ |
| [05e](notebooks/05_martingales/05e_doob_azuma.ipynb) | 선택 | 둡 부등식과 아주마 부등식 | Williams 14.6 Doob's submartingale inequality · G&S 4e 12.2 Martingale differences and Hoeffding's inequality (bin packing 예제 포함) | 90 | ☐ |
| [05r](notebooks/05_martingales/05r_review.ipynb) | 복습 | 모듈 04-05 복습 (닫힌 책) | Durrett 3e Ch.3 Renewal Processes (3.1 · Ross 12e Ch.7 Renewal Theory (renewal reward | 60 | ☐ |
| **캡스톤** | | | | | |
| [C3](notebooks/capstones/C3_mcmc_ising.ipynb) | 캡스톤 | 캡스톤 3: MCMC와 이징 모형 | Durrett 3e 1.6 (special examples: detailed balance · Lawler 2e Ch. 7 (reversible Markov chains | 150 | ☐ |
| **06 브라운 운동** | | | | | |
| [06a](notebooks/06_brownian/06a_random_walk_to_bm.ipynb) | 핵심 | 랜덤 워크에서 브라운 운동으로 | Lawler 2e 8.1 · Ross 12e 10.1 | 90 | ☐ |
| [06b](notebooks/06_brownian/06b_gaussian_process_quadratic_variation.ipynb) | 핵심 | 가우스 과정과 이차 변동 | Ross 12e 10.7 · Shreve II 3.3.2 | 90 | ☐ |
| [06c](notebooks/06_brownian/06c_reflection_max_hitting.ipynb) | 핵심 | 반사 원리, 최댓값, 도달 시간 | Ross 12e 10.2 · Shreve II 3.6 | 90 | ☐ |
| [06d](notebooks/06_brownian/06d_brownian_martingales_exit.ipynb) | 핵심 | 브라운 마팅게일과 탈출 문제 | Karlin–Taylor 7.5 · Ross 12e 10.2 | 90 | ☐ |
| [06e](notebooks/06_brownian/06e_bridge_drift_gbm.ipynb) | 핵심 | 브라운 다리, 드리프트, 기하 브라운 운동 | Lawler 2e 8.7 · Ross 12e 10.3 | 90 | ☐ |
| **07 이토 적분과 SDE** | | | | | |
| [07a](notebooks/07_sde/07a_ito_integral_isometry.ipynb) | 핵심 | 이토 적분과 등거리성 | Øksendal 3.1 (Itô 적분의 구성) · Shreve II 4.2 (단순 피적분함수의 Itô 적분) | 100 | ☐ |
| [07b](notebooks/07_sde/07b_ito_formula.ipynb) | 핵심 | 이토 공식 | Øksendal 4.1 (1차원 Itô 공식 · Shreve II 4.4 (Itô-Doeblin formula: 4.4.1-4.4.3 BM 용 | 100 | ☐ |
| [07c](notebooks/07_sde/07c_euler_maruyama_milstein.ipynb) | 핵심 | 오일러-마루야마와 밀스타인 | Øksendal 5.1 (예제·해법) · Higham 2001 §4 (Euler-Maruyama) | 100 | ☐ |
| [07d](notebooks/07_sde/07d_ornstein_uhlenbeck.ipynb) | 핵심 | 오른슈타인-울렌벡 과정 | Øksendal 5.1 (선형 SDE 의 해법 · Shreve II 4.4 (Example 4.4.10 Vasicek 금리모형 = OU) | 90 | ☐ |
| [07e](notebooks/07_sde/07e_gbm_black_scholes_feynman_kac.ipynb) | 핵심 | GBM, 블랙-숄즈, 파인만-카츠 | Shreve II 4.5 (Black-Scholes-Merton 방정식) · Øksendal 8.2 (Feynman-Kac 공식 | 110 | ☐ |
| **캡스톤** | | | | | |
| [C4](notebooks/capstones/C4_option_pricing.ipynb) | 캡스톤 | 캡스톤 4: 옵션 가격 결정 | Shreve II 4.5 (Black–Scholes–Merton) · Glasserman 4.1 (control variates) | 150 | ☐ |
| [C5](notebooks/capstones/C5_stochastic_sir.ipynb) | 캡스톤 | 캡스톤 5: 확률적 SIR 전염병 모형 | Durrett 3e 4.1 (CTMC definitions) · Ross 12e 5.2-5.3 (exponential races | 150 | ☐ |


## spkit 한눈에 보기

| 모듈 | 주요 함수 |
|---|---|
| `spkit` | `rng_for(id)`, `fast()`, `scale(n)`, `setup_plots()` |
| `spkit.mc` | `mc_estimate`, `mc_proportion`, `compare_table`, `assert_close` |
| `spkit.markov` | `simulate_chain`, `stationary`(+`_eig`, `_power`), `classify_states`, `absorption`, `hitting_times`, `tv_distance`, `gamblers_ruin_*`, `ehrenfest_matrix`, `metropolis_chain`, `pagerank`, `galton_watson`, `extinction_probability` |
| `spkit.poisson` | `poisson_arrivals`, `poisson_arrivals_conditional`, `nhpp_thinning`, `thin`, `count_in_bins`, `compound_poisson`, `age_residual_at` |
| `spkit.ctmc` | `gillespie`, `gillespie_paths`, `transition_matrix`, `stationary_ctmc`, `embedded_chain`, `uniformize`, `birth_death_*`, `mm1_stationary`, `mmc_metrics`, `time_average_occupation` |
| `spkit.renewal` | `renewal_times`, `renewal_count`, `renewal_reward`, `age_residual_process` |
| `spkit.brownian` | `brownian_paths`, `coarsen`, `brownian_bridge`, `gbm_paths`, `scaled_random_walk`, `running_max`, `first_passage_index`, `quadratic_variation` |
| `spkit.sde` | `euler_maruyama`, `milstein`, `ou_exact`, `black_scholes_call/put`, `mc_option_price` |
| `spkit.plots` | `plot_paths`, `plot_hist_vs_pmf`, `plot_hist_vs_pdf` |

노트북 안에서 세 번째로 복붙하게 되는 헬퍼가 다음 `spkit` 후보입니다.
