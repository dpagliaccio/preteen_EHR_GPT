# Clustered Bootstrapping of performance metrics
# David Pagliaccio
# Feb 17, 2026

#adapted from: https://github.com/pab2163/identifying_suicide_related_language/blob/95760fa7fc7fcb2d7186df4a831d309ec8eeaa4e/analysis_scripts/helper_functions.R#L109

# load tidyverse library
library(tidyverse)



# ============================================================
# DEFINE Function: evaluate_predictions()
# ============================================================
# Purpose:
#   Computes classification performance metrics for one or more
#   prediction columns in a dataframe, using caret::confusionMatrix.
#
# Arguments:
#   df : Data frame containing reference and prediction columns
#   reference_col       : Name of the ground-truth (reference) column
#   prediction_cols     : Character vector of prediction column names
#   positive_class      : Label for the positive class (default = 'Yes')
#
# Returns:
#   A data frame of evaluation metrics (e.g., Precision, Recall, F1)
#   for each prediction column.
# ============================================================
evaluate_predictions = function(df, reference_col, prediction_cols, positive_class = 'Yes') {
  
  # Check that the reference column exists
  if (!reference_col %in% names(df)) {
    stop(paste("Reference column '", reference_col, "' not found in the dataframe."))
  }
  
  # Check that all prediction columns exist
  missing_preds = setdiff(prediction_cols, names(df))
  if (length(missing_preds) > 0) {
    stop(paste("Prediction column(s) '", paste(missing_preds, collapse = ", "), "' not found in the dataframe."))
  }
  
  # Initialize list to store results
  results_list = list()
  
  # Loop through prediction columns
  for (pred_col in prediction_cols) {
    # Convert to factor for confusionMatrix
    predicted = factor(df[[pred_col]])
    reference = factor(df[[reference_col]])
    
    # Compute confusion matrix statistics
    cm = caret::confusionMatrix(
      data = predicted,
      reference = reference,
      positive = positive_class,
      mode = 'everything'
    )
    
    # Extract key metrics
    stats = data.frame(
      Sensitivity = cm$byClass[['Sensitivity']],
      Specificity = cm$byClass[['Specificity']],
      Pos_Pred_Value = cm$byClass[['Pos Pred Value']],
      Neg_Pred_Value = cm$byClass[['Neg Pred Value']],
      Precision = cm$byClass[['Precision']],
      Recall = cm$byClass[['Recall']],
      F1 = cm$byClass[['F1']],
      Prevalence = cm$byClass[['Prevalence']],
      Detection_Rate = cm$byClass[['Detection Rate']],
      Detection_Prevalence = cm$byClass[['Detection Prevalence']],
      Balanced_Accuracy = cm$byClass[['Balanced Accuracy']]
    )
    
    # Add prediction column name
    stats$PredictionColumn = pred_col
    
    # Store results
    results_list[[pred_col]] = stats
  }
  
  # Combine all results into a single data frame
  results_df = do.call(rbind, results_list)
  
  return(results_df)
}



# ============================================================
# DEFINE Function: bootstrap_evaluate_predictions()
# ============================================================
# Purpose: Performs bootstrap resampling to estimate confidence intervals
#          for prediction performance metrics. Samples IDs (not rows) with
#          replacement to preserve within-participant correlation structure.
#
# Arguments:
#   full_df         - Data frame containing reference and prediction columns
#   id_col          - String: name of column containing participant IDs
#   reference_col   - String: name of column with true/reference labels
#   prediction_cols - Character vector: names of columns with predictions to evaluate
#   positive_class  - String: which class is considered "positive" (default: 'Yes')
#   n_iterations    - Integer: number of bootstrap iterations (default: 1000)
#   seed            - Integer: random seed for reproducibility (default: 123)
#   alpha           - Numeric: significance level for CI calculation (default: 0.05 for 95% CI)
#
# Returns: Data frame with columns:
#   - Metric: performance metric name (e.g., Accuracy, Sensitivity, etc.)
#   - Value: mean metric value across bootstrap iterations
#   - CI_Lower: lower bound of (1-alpha)% confidence interval
#   - CI_Upper: upper bound of (1-alpha)% confidence interval
#   - PredictionColumn: which prediction column the metrics apply to
# ============================================================
bootstrap_evaluate_predictions = function(
    full_df,
    id_col,
    reference_col,
    prediction_cols,  
    positive_class = 'Yes',
    n_iterations = 1000,
    seed = 123,
    alpha=0.05
) {
  set.seed(seed)
  
  bootstrap_results = list()
  unique_ids = unique(full_df[[id_col]])
  
  for (i in 1:n_iterations) {
    print(i)
    # Sample IDs with replacement
    boot_ids = sample(unique_ids, length(unique_ids), replace = TRUE)
    
    # Get all observations for sampled IDs
    boot_sample = lapply(boot_ids, function(id) {
      full_df[full_df[[id_col]] == id, ]
    })
    boot_sample = do.call(rbind, boot_sample)
    
    # Evaluate predictions
    eval_result = evaluate_predictions(
      boot_sample,
      reference_col = reference_col,
      prediction_cols = prediction_cols,
      positive_class = positive_class
    )
    
    eval_result$BootstrapIteration = i
    bootstrap_results[[i]] = eval_result
  }
  
  # Combine all bootstrap iterations
  results_df = do.call(rbind, bootstrap_results)
  
  # Calculate CIs by prediction column (raw values)
  metric_cols = setdiff(names(results_df), c("PredictionColumn", "BootstrapIteration"))
  
  summary_list = list()
  
  for (pred_col in prediction_cols) {
    pred_data = results_df[results_df$PredictionColumn == pred_col, ]
    
    summary_list[[pred_col]] = data.frame(
      Metric = metric_cols,
      Value = sapply(metric_cols, function(col) mean(pred_data[[col]], na.rm = TRUE)),
      CI_Lower = sapply(metric_cols, function(col) quantile(pred_data[[col]], alpha/2, na.rm = TRUE)),
      CI_Upper = sapply(metric_cols, function(col) quantile(pred_data[[col]], 1-alpha/2, na.rm = TRUE)),
      PredictionColumn = pred_col,
      row.names = NULL
    )
  }
  
  # Combine into single dataframe
  final_df = do.call(rbind, summary_list)
  rownames(final_df) = NULL
  
  return(final_df)
}



### RUN BOOTSTRAPPING
# load output of LLM
df <- read.csv("gpt_output.csv",sep = ",")
#select columns
df <- df %>% select(ID,Concensus,GPT,Regex) 
#create either/or from concensus | regex
df$Either <- ifelse(df$GPT=="Yes"|df$Regex=="Yes","Yes","No")

#run bootstrapping, 1000 iterations, 95% CI
out_boot <- bootstrap_evaluate_predictions(full_df = df,
  reference_col="Concensus",
  prediction_col=c("GPT","Regex","Either"),
  positive_class = 'Yes', id_col = "ID", 
  n_iterations = 1000, seed = 123, alpha=0.05)

#select and format output 
out_boot %>% 
  filter(Metric %in% c("Specificity", "Sensitivity", "Precision", "F1")) %>%
  mutate(
    Result = sprintf("%.2f%% [%.2f - %.2f]", 
                     Value * 100, 
                     CI_Lower * 100, 
                     CI_Upper * 100)
  ) %>% select(PredictionColumn, Metric, Result) %>%tidyr::pivot_wider(names_from = PredictionColumn, values_from = Result) %>%
  slice(match(c("Specificity", "Sensitivity", "Precision", "F1"), Metric))
  
  
