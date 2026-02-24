# ============================
# Testlet Response Theory (TRT) quick analysis
# ============================
# Pipeline:
#  1) Load + preprocess
#  2) Train/test split
#  3) Fit TRT with brms (random intercepts: student, item, testlet)
#  4) Evaluate (ROC-AUC) + Person fit (infit/outfit)
#  5) Extract random effects and make a few plots

# ----Load Packages ----
library(dplyr)
library(tidyr)
library(ggplot2)
library(brms)
library(pROC)

# ---- Params----
DATA_PATH <- "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/resolve/main/codeinsights_llm_simulation/data/code_insights_clean.csv"
RANDOM_SEED        <- 123
TEST_FRACTION      <- 0.20
N_TESTLET_GROUPS   <- 3     # split testlets into N groups, take the first group
N_STUDENT_GROUPS   <- 10    # split students into N groups, take the first group
USE_ALL_TESTLETS   <- FALSE # set TRUE to skip testlet subsetting - Data is too big so subset is recommended for quick run
USE_ALL_STUDENTS   <- FALSE # set TRUE to skip student subsetting - Data is too big so subset is recommended for quick run
BAYES_ITER         <- 2000
BAYES_CHAINS       <- 2
BAYES_CORES        <- 2
ADAPT_DELTA        <- 0.95

set.seed(RANDOM_SEED)

# ============================
# Helper Functions
# ============================

load_data <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

split_testlet_item <- function(df) {
  df %>%
    separate(ItemID_SF, into = c("Testlet_ID", "Item_Within_Testlet"),
             sep = "_", remove = FALSE)
}

keep_latest_attempt <- function(df) {
  df %>%
    group_by(StudentID_SF, ItemID_SF) %>%
    filter(T == max(T)) %>%
    ungroup()
}

drop_no_variation_items <- function(df_latest) {
  wide <- df_latest %>%
    select(StudentID_SF, ItemID_SF, ItemScore) %>%
    pivot_wider(names_from = ItemID_SF, values_from = ItemScore)
  
  score_matrix <- as.matrix(wide[, -1, drop = FALSE])
  df_items <- as.data.frame(score_matrix)
  unique_counts <- apply(df_items, 2, function(x) length(unique(x[!is.na(x)])))
  keep_cols <- names(unique_counts)[unique_counts >= 2]
  
  list(
    data_wide_clean = wide[, c("StudentID_SF", keep_cols), drop = FALSE],
    kept_item_ids   = keep_cols
  )
}

subset_by_testlet_student <- function(df, use_all_testlets = FALSE, use_all_students = FALSE,
                                      n_testlet_groups = 3, n_student_groups = 10) {
  out <- df
  
  if (!use_all_testlets) {
    testlet_ids <- unique(out$Testlet_ID)
    testlet_split <- split(sample(testlet_ids),
                           cut(seq_along(testlet_ids), breaks = n_testlet_groups, labels = FALSE))
    first_testlet_ids <- testlet_split[[1]]
    out <- out %>% filter(Testlet_ID %in% first_testlet_ids)
  }
  
  if (!use_all_students) {
    student_ids <- unique(out$StudentID_SF)
    student_split <- split(sample(student_ids),
                           cut(seq_along(student_ids), breaks = n_student_groups, labels = FALSE))
    first_student_ids <- student_split[[1]]
    out <- out %>% filter(StudentID_SF %in% first_student_ids)
  }
  
  out
}

first_attempt_only <- function(df) {
  df %>%
    group_by(StudentID_SF, ItemID_SF) %>%
    slice_min(order_by = T, n = 1, with_ties = FALSE) %>%
    ungroup()
}

train_test_split <- function(df, test_fraction = 0.2) {
  df <- df %>% mutate(split = ifelse(runif(n()) < test_fraction, "test", "train"))
  list(
    train = df %>% filter(split == "train"),
    test  = df %>% filter(split == "test")
  )
}

fit_trt <- function(train_df, iter = 2000, chains = 2, cores = 2, adapt_delta = 0.95) {
  brm(
    ItemScore ~ 1 + (1 | StudentID_SF) + (1 | ItemID_SF) + (1 | Testlet_ID),
    data   = train_df,
    family = bernoulli(),
    chains = chains,
    cores  = cores,
    iter   = iter,
    control = list(adapt_delta = adapt_delta)
  )
}

eval_auc <- function(fit, test_df) {
  post_prob <- posterior_epred(fit, newdata = test_df, allow_new_levels = TRUE)
  pred_mean <- colMeans(post_prob)
  roc_obj   <- roc(response = test_df$ItemScore, predictor = pred_mean)
  list(roc = roc_obj, auc = as.numeric(auc(roc_obj)))
}

compute_person_fit <- function(fit, train_df) {
  post_prob_mat <- posterior_linpred(fit, newdata = train_df, transform = TRUE)
  p_hat <- colMeans(post_prob_mat)
  y     <- train_df$ItemScore
  
  res_z <- (y - p_hat) / sqrt(p_hat * (1 - p_hat))
  w     <- p_hat * (1 - p_hat)
  
  train_df %>%
    transmute(StudentID_SF, res2 = res_z^2, w = w) %>%
    group_by(StudentID_SF) %>%
    summarise(
      n_items = n(),
      outfit  = mean(res2),
      infit   = sum(w * res2) / sum(w),
      .groups = "drop"
    ) %>%
    arrange(desc(outfit))
}

extract_random_effects <- function(fit) {
  # All are arrays with columns like Estimate, Est.Error, Q2.5, Q97.5
  abil_arr  <- ranef(fit)$StudentID_SF[, , "Intercept"]
  item_arr  <- ranef(fit)$ItemID_SF   [, , "Intercept"]
  test_arr  <- ranef(fit)$Testlet_ID  [, , "Intercept"]
  
  ability_df <- as.data.frame(abil_arr)
  ability_df$StudentID_SF <- rownames(abil_arr)
  
  item_df <- as.data.frame(item_arr)
  item_df$ItemID_SF <- rownames(item_arr)
  
  testlet_df <- as.data.frame(test_arr)
  testlet_df$Testlet_ID <- rownames(test_arr)
  
  list(ability = ability_df, item = item_df, testlet = testlet_df)
}

plot_roc <- function(roc_obj, auc_value) {
  plot(roc_obj, main = paste0("ROC Curve (AUC = ", round(auc_value, 3), ")"))
}

plot_testlet_vs_correctness <- function(testlet_df, first_data) {
  correctness_per_testlet <- first_data %>%
    group_by(Testlet_ID) %>%
    summarise(correctness = mean(ItemScore, na.rm = TRUE), .groups = "drop")
  
  merged <- left_join(testlet_df, correctness_per_testlet, by = "Testlet_ID")
  
  ggplot(merged, aes(x = Estimate, y = correctness)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    labs(
      title = "Testlet random effect vs. average correctness",
      x = "Testlet RE (logit)",
      y = "Mean correctness"
    ) +
    theme_minimal(base_size = 13)
}

plot_item_difficulties_by_testlet <- function(item_difficulty_by_testlet) {
  means <- item_difficulty_by_testlet %>%
    group_by(Testlet_ID) %>%
    summarise(mean_difficulty = mean(Estimate, na.rm = TRUE), .groups = "drop")
  
  plot_df <- left_join(item_difficulty_by_testlet, means, by = "Testlet_ID")
  
  ggplot(plot_df, aes(x = Estimate, y = reorder(Testlet_ID, mean_difficulty))) +
    geom_point(size = 2, alpha = 0.7) +
    labs(
      title = "Item difficulties by testlet",
      x = "Item difficulty (logit)",
      y = "Testlet ID"
    ) +
    theme_minimal(base_size = 13)
}

summarise_testlet_ranges <- function(item_difficulty_by_testlet, top_n = 5) {
  testlet_ranges <- item_difficulty_by_testlet %>%
    group_by(Testlet_ID) %>%
    summarise(
      range_difficulty = max(Estimate) - min(Estimate),
      n_items = n(),
      .groups = "drop"
    ) %>%
    filter(n_items > 1)
  
  list(
    top    = testlet_ranges %>% arrange(desc(range_difficulty)) %>% slice(1:top_n),
    bottom = testlet_ranges %>% arrange(range_difficulty)         %>% slice(1:top_n)
  )
}

# ============================
# Main
# ============================

# 1) Load + Basic Preprocessing
data_raw <- load_data(DATA_PATH)
data_raw <- split_testlet_item(data_raw)
data_last <- keep_latest_attempt(data_raw)

# Drop items with no variation (kept list used downstream if you want to filter)
var_drop <- drop_no_variation_items(data_last)
kept_item_ids <- var_drop$kept_item_ids

# 2) Optional subsetting for a fast run
first_data <- subset_by_testlet_student(
  df = data_raw,
  use_all_testlets = USE_ALL_TESTLETS,
  use_all_students = USE_ALL_STUDENTS,
  n_testlet_groups = N_TESTLET_GROUPS,
  n_student_groups = N_STUDENT_GROUPS
)

# Train/test split on first attempts only
first_attempts <- first_attempt_only(first_data)
splits <- train_test_split(first_attempts, test_fraction = TEST_FRACTION)
train_data <- splits$train
test_data  <- splits$test

# 3) Fit TRT
fit <- fit_trt(
  train_data,
  iter = BAYES_ITER,
  chains = BAYES_CHAINS,
  cores = BAYES_CORES,
  adapt_delta = ADAPT_DELTA
)

# 4) Evaluate (ROC–AUC) + Person fit
auc_out <- eval_auc(fit, test_data)
plot_roc(auc_out$roc, auc_out$auc)
cat(sprintf("AUC: %.3f\n", auc_out$auc))

person_fit <- compute_person_fit(fit, train_data)

# 5) Random Effects
re <- extract_random_effects(fit)
ability_df <- re$ability
item_df    <- re$item
testlet_df <- re$testlet
# Map items to testlets present in the (possibly subset) first_data
item_map <- first_data %>%
  select(ItemID_SF, Testlet_ID) %>%
  distinct()

# Combine item + testlet for a testlet-adjusted item difficulty
combined_df <- item_df %>%
  left_join(item_map, by = "ItemID_SF") %>%
  left_join(testlet_df, by = "Testlet_ID", suffix = c("_item", "_testlet")) %>%
  mutate(total_difficulty = Estimate_item + Estimate_testlet) %>%
  select(ItemID_SF, Testlet_ID, Estimate_item, Estimate_testlet, total_difficulty)

# For per-testlet item difficulty views, define a simple difficulty_df from item REs
difficulty_df <- item_df %>%
  transmute(ItemID_SF, Estimate)

item_difficulty_by_testlet <- difficulty_df %>%
  left_join(item_map, by = "ItemID_SF") %>%
  filter(!is.na(Testlet_ID))

# 6) Plots
plot_testlet_vs_correctness(testlet_df, first_data)
plot_item_difficulties_by_testlet(item_difficulty_by_testlet)

# ============================
# End
# ============================