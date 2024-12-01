# Load required libraries
library(tidyverse)
library(lme4)
library(lubridate)

# Set the theme for plots
theme_set(
  theme_classic() + #set the theme 
    theme(text = element_text(family = "Times", size = 18)) #set the default text size
)

# Set the working directory
# setwd("/path/to/parent/folder")

# -----------------------------------
# Functions Definition
# -----------------------------------

# Convert 'time_delta' to seconds
convert_to_seconds <- function(time_string) {
  parts <- strsplit(time_string, " days ")[[1]]
  days <- as.numeric(parts[1])
  time <- if (length(parts) > 1) lubridate::hms(parts[2]) else lubridate::hms("00:00:00")
  days * 24 * 3600 + as.numeric(time)
}

# Calculate percentage of time deltas over a threshold
calculate_percentage_over_hours <- function(data, threshold) {
  percentage <- 100 * nrow(data %>% filter(time_delta_seconds > threshold * 60 * 60)) / 
                nrow(data %>% filter(!is.na(time_delta_seconds)))
  percentage
}

# Create scatter plots
create_scatterplot <- function(df, x_var, y_var, color_var = NULL, title) {
  p <- ggplot(df, aes(x = !!sym(x_var), y = !!sym(y_var))) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
    labs(title = title, x = gsub("_", " ", str_to_title(x_var)), y = gsub("_", " ", str_to_title(y_var))) +
    theme_minimal()
  if (!is.null(color_var)) {
    p <- p + aes(color = !!sym(color_var)) + scale_color_viridis_c() + labs(color = gsub("_", " ", str_to_title(color_var)))
  }
  if (y_var == "median_time_delta_seconds") {
    p <- p + scale_y_log10()
  }
  p
}

# -----------------------------------
# Data Import
# -----------------------------------

# Reading CSV files
csv_files <- list.files(path = "csv_file", pattern = "*.csv", full.names = TRUE)
filtered_files <- list.files(path = "filtered_csv_file", pattern = "*.csv", full.names = TRUE)
data_frames <- lapply(csv_files, read.csv)
filtered_data_frames <- lapply(filtered_files, read.csv)
names(data_frames) <- tools::file_path_sans_ext(basename(csv_files))
names(filtered_data_frames) <- tools::file_path_sans_ext(basename(filtered_files))

# -----------------------------------
# Data Processing
# -----------------------------------

# Apply conversion to all data frames
for (i in seq_along(data_frames)) {
  if ("time_delta" %in% names(data_frames[[i]])) {
    data_frames[[i]]$time_delta_seconds <- sapply(data_frames[[i]]$time_delta, convert_to_seconds)
  } else {
    warning(paste("time_delta column not found in", names(data_frames)[i]))
  }
}

for (i in seq_along(filtered_data_frames)) {
  if ("time_delta" %in% names(filtered_data_frames[[i]])) {
    filtered_data_frames[[i]]$time_delta_seconds <- sapply(filtered_data_frames[[i]]$time_delta, convert_to_seconds)
  } else {
    warning(paste("time_delta column not found in", names(filtered_data_frames)[i]))
  }
}

# -----------------------------------
# Data Analysis
# -----------------------------------

# Combine time_delta_seconds from all data frames for analysis
all_time_delta_seconds <- do.call(rbind, lapply(data_frames, function(df) {
  data.frame(time_delta_seconds = df$time_delta_seconds, source = rep(deparse(substitute(df)), nrow(df)))
}))

# Statistical summary
time_delta_stats <- all_time_delta_seconds %>%
  mutate(time_delta_minutes = time_delta_seconds / 60) %>%
  summarise(
    Mean = mean(time_delta_minutes, na.rm = TRUE),
    Median = median(time_delta_minutes, na.rm = TRUE),
    Min = min(time_delta_minutes, na.rm = TRUE),
    Max = max(time_delta_minutes, na.rm = TRUE),
    `95th Percentile` = quantile(time_delta_minutes, 0.95, na.rm = TRUE)
  )
print(time_delta_stats)

# Percentage calculations over various hour thresholds
cat("Percentage longer than 1 hour:", round(calculate_percentage_over_hours(all_time_delta_seconds, 1), 2), "%\n")
cat("Percentage longer than 2 hours:", round(calculate_percentage_over_hours(all_time_delta_seconds, 2), 2), "%\n")
cat("Percentage longer than 3 hours:", round(calculate_percentage_over_hours(all_time_delta_seconds, 3), 2), "%\n")

# Density plot of time delta
p <- ggplot(all_time_delta_seconds, aes(x = time_delta_seconds / 60)) +
  geom_density(fill = "skyblue", color = "darkblue", alpha = 0.7) +
  scale_x_log10(
    breaks = c(0.1, 1, 10, 100, 1000, 10000),
    labels = c("0.1", "1", "10", "100", "1000", "10000"),
    name = "Minutes (log scale)"
  ) +
  labs(title = "Density Plot of log(time_delta)", y = "Density") +
  theme_minimal() +
  geom_vline(aes(xintercept = median_minutes), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = mean_minutes), color = "blue", linetype = "dashed") +
  annotate("text", x = median_minutes, y = Inf, label = paste("Median (", median_minutes, " mins)"), color = "red", vjust = 2, hjust = -0.1) +
  annotate("text", x = mean_minutes, y = Inf, label = paste("Mean (", mean_minutes, " mins)"), color = "blue", vjust = 2, hjust = 1.1)
ggsave("log_time_delta_density_plot.png", p, width = 10, height = 6)

# Action count analysis
for (name in names(data_frames)) {
  cat("Dataframe:", name, "\n")
  print(table(data_frames[[name]]$action))
  cat("\n")
}

# -----------------------------------
# Filtered Data Analysis
# -----------------------------------

# Combine filtered time_delta_seconds for more focused analysis
filtered_all_time_delta_seconds <- do.call(rbind, lapply(filtered_data_frames, function(df) {
  data.frame(time_delta_seconds = df$time_delta_seconds, source = rep(deparse(substitute(df)), nrow(df)))
}))

# Descriptive statistics for filtered data
filtered_time_delta_stats <- filtered_all_time_delta_seconds %>%
  mutate(time_delta_minutes = time_delta_seconds / 60) %>%
  summarise(
    Mean = mean(time_delta_minutes, na.rm = TRUE),
    Median = median(time_delta_minutes, na.rm = TRUE),
    Min = min(time_delta_minutes, na.rm = TRUE),
    Max = max(time_delta_minutes, na.rm = TRUE),
    `95th Percentile` = quantile(time_delta_minutes, 0.95, na.rm = TRUE)
  )
print(filtered_time_delta_stats)

# Percentage calculations for filtered data
cat("Percentage of filtered time delta longer than 1 hour:", round(calculate_percentage_over_hours(filtered_all_time_delta_seconds, 1), 2), "%\n")
cat("Percentage of filtered time delta longer than 2 hours:", round(calculate_percentage_over_hours(filtered_all_time_delta_seconds, 2), 2), "%\n")
cat("Percentage of filtered time delta longer than 3 hours:", round(calculate_percentage_over_hours(filtered_all_time_delta_seconds, 3), 2), "%\n")

# Action count for filtered data
for (name in names(filtered_data_frames)) {
  cat("Dataframe (filtered):", name, "\n")
  print(table(filtered_data_frames[[name]]$action))
  cat("\n")
}

# -----------------------------------
# Research Questions Analysis
# -----------------------------------

# Function to analyze dataframes based on specific criteria
analyze_dataframe <- function(df) {
  df %>%
    group_by(id) %>%
    summarise(
      median_time_delta_seconds = median(time_delta_seconds, na.rm = TRUE),
      total_submits = sum(action == "Submit", na.rm = TRUE),
      final_score = last(marks[action == "Submit"], default = NA)
    ) %>%
    ungroup()
}

# Apply analysis and plotting functions to filtered dataframes
for (name in names(filtered_data_frames)) {
  df_analysis_results <- analyze_dataframe(filtered_data_frames[[name]])
  
  # Submits vs Pause Duration plot
  plot1 <- create_scatterplot(df_analysis_results, "total_submits", "median_time_delta_seconds", "final_score", paste("Submits vs Pause Duration for", name))
  print(plot1)
  ggsave(paste0("scatterplot_submits_pause_", name, ".png"), plot1, width = 10, height = 6)
  
  # Submits vs Final Score plot
  plot2 <- create_scatterplot(df_analysis_results, "total_submits", "final_score", NULL, paste("Submits vs Final Score for", name))
  print(plot2)
  ggsave(paste0("scatterplot_submits_score_", name, ".png"), plot2, width = 10, height = 6)
}

# -----------------------------------
# Top Submitters Analysis
# -----------------------------------

# Function to get top submitters
get_top_submitters <- function(df, n = 3) {
  df %>%
    arrange(desc(total_submits)) %>%
    slice_head(n = n) %>%
    select(id, total_submits, final_score)
}

# Analyze top submitters across all dataframes
all_top_submitters <- data.frame()
for (name in names(filtered_data_frames)) {
  df <- filtered_data_frames[[name]]
  result <- analyze_dataframe(df)
  top_submitters <- get_top_submitters(result)
  top_submitters$dataframe <- name
  all_top_submitters <- bind_rows(all_top_submitters, top_submitters)
  print(top_submitters)
}

# Analyze for repeated IDs among top submitters
repeated_ids <- all_top_submitters %>%
  group_by(id) %>%
  summarise(
    frequency = n(),
    dataframes = paste(dataframe, collapse = ", "),
    total_submits = paste(total_submits, collapse = ", "),
    final_scores = paste(final_score, collapse = ", ")
  ) %>%
  filter(frequency > 1) %>%
  arrange(desc(frequency))
cat("IDs appearing in top submitters more than once:\n")
print(repeated_ids)