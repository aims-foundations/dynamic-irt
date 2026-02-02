+++
title = "Dynamics of Learning"
date = "2024-11-10"
outputs = ["Reveal"]
[logo]
src = "images/sail-logo.jpg"
+++

{{< slide auto-animate="" >}}
## Dynamics of Learning

<div style="text-align:center;">
Sang Truong, Duc Nguyen, Tho Quan, Sanmi Koyejo, Nick Habber
</div>

<div style="text-align:center; font-style:italic;">
Work in Progress
</div>

---
{{< slide auto-animate="" >}}
### 1. Introduction
**Problem:** How can we model the student's dynamics of learning, especially in coding when the student's ability changes over time?

**Potential contributions:**  
- Provide an open-source dataset of coding questions and student answers for the community.  
- Propose an approach for modeling students' abilities and test case difficulties based on Dynamic Item Response Theory (IRT).  
- Propose a method for simulating student behaviors based on large language models and latent dynamical models.

---
{{< slide auto-animate="" >}}
### 2. Data
- 2 courses: Programming Fundamentals (PF) and Data Structures and Algorithms (DSA)
- 2 quaters for each course (PF222, PF232, DSA221, DSA231)
- 3286 students in total

---
{{< slide auto-animate="" >}}
### 2. Data
#### Coding Data - Questions
<img src="figures/question_table.png" style="width: 100%;">


---
{{< slide auto-animate="" >}}
### 2. Data
#### Coding Data - Student answers
<img src="figures/response_table.png" style="width: 100%;">


---
{{< slide auto-animate="" >}}
### 3. Naive Baselines

(Not Yet Decided)

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

**Students:**
{{% fragment %}} 
-  There are $N$ students, each is indexed by $n \in [N]$.
-  Each student has $T_n = \\{ t_{n1}, \cdots, t_{nT} \\}$ submission times.
-  Each submission time $t$ is modeled with a $\theta_t$, representing the student ability at that time.
{{% /fragment %}}

**Questions:**
{{% fragment %}} 
-   There are $Q$ questions, each question has 5 to 15 testcases.
-   The set of all testcases is $M$. Each element is indexed by $m \in [M]$
-   Each testcase is modeled with a $z_m$ indicating the testcase difficulty.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

-   At each submission time $t$, the student submits an answer to a question $q \in Q$.
-   She will receive the scores of all testcases $m \in M_q$ of that question: $Y_{nt} = \\{ y_{ntm} \\}_{i=1}^{|M_q|}$.

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
{{% fragment %}}For the duration of the course, the score of a student $n$ at time $t$ on testcase $m$ is modeled as:{{% /fragment %}}

{{% fragment %}}
```math
p(y|\theta_t, z_m) = \sigma(\theta_t + z_m)
```
where $\sigma$ is the sigmoid function.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

{{% fragment %}}We assume that the student ability $\theta_{1:T}$ of a student is drawn from a Gaussian Process{{% /fragment %}}
{{% fragment %}}and the covariance matrix is given by the kernel function $K$ with hyperparameters $\psi_K$.{{% /fragment %}}

{{% fragment %}}
```math
\theta_{1:T} \sim \mathcal{G}\mathcal{P}(\mathbf{0}, \Sigma; \psi_K)
```
{{% /fragment %}}

{{% fragment %}}
```math
\Sigma = K(T, T; \psi_K)
```
{{% /fragment %}}

<img src="figures/GP.gif">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
{{% fragment %}}We assume each testcase is independent and the difficulty $z_m$ of a testcase is drawn from a Normal distribution.{{% /fragment %}}

{{% fragment %}}
```math
z_m \sim \mathcal{N}(0, 1)
```
{{% /fragment %}}

<!-- {{% fragment %}} -->
<!-- *Open question: Shoule we use a GP for the testcase difficulty as well?*{{% /fragment %}} -->

<!-- ---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
{{% fragment %}}
With a pre-defined $K$ and $\psi_K$, we can compute the posterior distribution of the student ability $\theta_t$ given the observed data $Y_{nt}$.
{{% /fragment %}}

{{% fragment %}}
Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;
{{% /fragment %}}

**Student 0:**

{{% fragment %}}
<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/student_0_theta_dist.png" style="width: 40%;">
<img src="figures/dsa_hk231_seed45_npoints500_kernelRBF_lengthscale1.0/student_0_theta_dist.png" style="width: 40%;">
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
With a pre-defined $K$ and $\psi_K$, we can compute the posterior distribution of the student ability $\theta_t$ given the observed data $Y_{nt}$.

Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;

**Student 1:**

<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/student_1_theta_dist.png" style="width: 40%;">
<img src="figures/dsa_hk231_seed45_npoints500_kernelRBF_lengthscale1.0/student_1_theta_dist.png" style="width: 40%;">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
With a pre-defined $K$ and $\psi_K$, we can compute the posterior distribution of the student ability $\theta_t$ given the observed data $Y_{nt}$.

Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;

**Student 2:**

<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/student_2_theta_dist.png" style="width: 40%;">
<img src="figures/dsa_hk231_seed45_npoints500_kernelRBF_lengthscale1.0/student_2_theta_dist.png" style="width: 40%;">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
With a pre-defined $K$ and $\psi_K$, we can compute the posterior distribution of the student ability $\theta_t$ given the observed data $Y_{nt}$.

Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;

**Student 3:**

<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/student_3_theta_dist.png" style="width: 40%;">
<img src="figures/dsa_hk231_seed45_npoints500_kernelRBF_lengthscale1.0/student_3_theta_dist.png" style="width: 40%;">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
With a pre-defined $K$ and $\psi_K$, we can compute the posterior distribution of the student ability $\theta_t$ given the observed data $Y_{nt}$.

Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;

**Student 4:**

<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/student_4_theta_dist.png" style="width: 40%;">
<img src="figures/dsa_hk231_seed45_npoints500_kernelRBF_lengthscale1.0/student_4_theta_dist.png" style="width: 40%;"> -->


---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
#### Hamiltonian Monte Carlo algorithm

*Conceptual pipeline:*
- Sample all the student abilities $\theta_{n, 1:T} \sim \mathcal{G}\mathcal{P}(\mathbf{0}, \Sigma_n; \psi_K)$
- Sample all the testcase difficulties $z_m \sim \mathcal{N}(0, 1)$
- Compute the probability of each student's answer $p(y_{ntm}|\theta_{nt}, z_m)$
- Hamiltonian Monte Carlo algorithm will decide to accept or reject the new sample based on the Hamiltonian dynamics.

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
#### Compute the Hamiltonian
The Hamiltonian $H(\theta, r)$ combines:
$H(\theta, r) = U(\theta) + K(r)$

-   **Potential Energy**: $U(\theta) = -\log p(\theta)$
-   **Kinetic Energy**: $K(r) = \frac{1}{2} r^\top M^{-1} r$

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT
#### Metropolis Acceptance Step
- Compute the new Hamiltonian $H(\theta^\*, r^\*)$.
- Accept the new state $(\theta^\*, r^\*)$ with probability:

```math
\alpha = \min\left(1, \exp(H(\theta_t, r_t) - H(\theta^*, r^*))\right)
```

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

Result (average Accuracy) for different RBF kernel with $\psi_K = \\{length\\_scale = 0.1, 0.5, 1\\}$;

<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale0.1/all_students_accuracy_hist.png" style="width: 32%;">
<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale0.5/all_students_accuracy_hist.png" style="width: 32%;">
<img src="figures/dsa_hk231_seed42_npoints500_kernelRBF_lengthscale1.0/all_students_accuracy_hist.png" style="width: 32%;">


---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

Result (average Accuracy) for different Matern kernel with $\psi_K = \\{\nu = 0.5, 1.5, 2.5\\}$;

<img src="figures/dsa_hk231_seed42_npoints500_kernelMatern_nu0.5/all_students_accuracy_hist.png" style="width: 32%;">
<img src="figures/dsa_hk231_seed42_npoints500_kernelMatern_nu1.5/all_students_accuracy_hist.png" style="width: 32%;">
<img src="figures/dsa_hk231_seed42_npoints500_kernelMatern_nu2.5/all_students_accuracy_hist.png" style="width: 32%;">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

$D_{KL}$ between the posterior distribution of the question difficulty $z_m$ in different runs.
Setting: $K$ is the RBF kernel with $\psi_K = \\{length\\_scale = 1\\}$; Seeds: 42, 45;

<img src="figures/kl_divergence_histogram.png" style="width: 50%;">

---
{{< slide auto-animate="" >}}
### 3. Dynamic IRT

Distribution of $z_m$ for different testcase $m$; The higher $z$ means the testcase is easier

<img src="figures/z_histogram_index_185.png" style="width: 40%;">
<img src="figures/z_histogram_index_350.png" style="width: 40%;">
<img src="figures/z_histogram_index_496.png" style="width: 40%;">
<img src="figures/z_histogram_index_882.png" style="width: 40%;">

---
{{< slide auto-animate="" >}}
### 4. Reccurent State Space Model

Recurrent model: $h_t = f_\phi(h_{t-1}, z_{t-1}, q_{t})$

Representation model: $z_t             \sim     q_{\phi}(z_t | e_t)$

Transition predictor: $\hat{z}\_t       \sim  p_{\phi}(\hat{z}_t | h_t)$

Embedding predictor: $\hat{e}\_t       \sim  p_{\phi}(\hat{e}_t | z_t)$

Score predictor: $\hat{r}\_t       \sim  p\_{\phi}(\hat{r}\_t | q_t, w_{q_t}, e_t)$

Text decoder: $\hat{x}\_t       \sim  p_{\psi}(\hat{x}_t | \hat{e}_t)$


---
{{< slide auto-animate="" >}}
### 4. Reccurent State Space Model
<img src="figures/latent_rssm.png" >

---
{{< slide auto-animate="" >}}
### 4. Reccurent State Space Model without Latent
Recurrent model: $h_t = f_\phi(h_{t-1}, e_{t-1}, q_{t})$

Embedding predictor: $\hat{e}\_t       \sim  p_{\phi}(\hat{e}_t | h_t)$

Score predictor: $\hat{r}\_t       \sim  p\_{\phi}(\hat{r}\_t | q_t, w_{q_t}, e_t)$

Text decoder: $\hat{x}\_t       \sim  p_{\psi}(\hat{x}_t | \hat{e}_t)$


---
{{< slide auto-animate="" >}}
### 4. Reccurent State Space Model without Latent
<img src="figures/rssm_wo_latent.png" >

---
{{< slide auto-animate="" >}}
### 5. Language Model Simulation


---
{{< slide auto-animate="" >}}
### 6. Conclusion
