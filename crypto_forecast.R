#
# clear the environment
#
  rm(list=ls(all.names = TRUE))
#
# load library for keras intefavce to Tensorflow
#
  library(keras)
#
# scales has functions to format dates in charts
#
  library(scales)
#
# rjson lets us convert model json to R to exract info
#
  library(rjson)
#
# RJSONIO gets around a bug in rjson--order of loading is important
#
  library(RJSONIO)
#
# zoo is used for smoothing (rollmean function)
#
  library(zoo)
#
# dplyr for pipes
#
  library(dplyr)
#
# ggplot for plotting results
#
  library(ggplot2)
#
  get_metrics <- function(predictions, labels) {
    predictions[predictions < 0.5] <- 0
    predictions[predictions > 0.5] <- 1
    predictions[predictions == 0.5] <-
      sample(c(0, 1), 
             length(predictions[predictions == 0.5]),
             replace = TRUE)
    accuracy <- length(which(predictions == labels)) /
      length(labels)
    tp <- length(which(predictions == labels &
                         predictions == 1))
    fp <- length(which(predictions != labels &
                         predictions == 1))
    tn <- length(which(predictions == labels &
                         predictions == 0))
    fn <- length(which(predictions != labels &
                         predictions == 0))
    if ((tp + fp) > 0) {
      precision <- tp / (tp + fp)
    } else {
      precision <- 0
    }
    if ((tp + fn) > 0) {
      recall <- tp / (tp + fn)
    } else {
      recall <- 0
    }
    if ((precision + recall) > 0) {
      f1 <- 2 * (precision * recall) / (precision + recall)
    } else {
      f1 <- 0
    }
    metrics <- list(predictions = predictions, 
                    accuracy = accuracy, 
                    precision = precision, 
                    recall = recall,
                    f1 = f1)
    return(metrics)
  }
#
#
#
#
# end function get_metrics
#  
#
#
#  
  make_layers_numeric <- function(experiment_records, mod_layers) {
#
# this section converts all layer text to values to make plotting easier
#
    for (i in 1:nrow(experiment_records)) {
      for (j in 1:length(mod_layers)) {
        column_index <- j + which(colnames(experiment_records) == "layer_1") - 1
        if (!(is.na(experiment_records[i, column_index]))) {
          if (is.na(as.integer(experiment_records[i, column_index]))) {
            if ((regexpr("dense", experiment_records[i, column_index], 1) > 0)) {
              pos_1 <- regexpr(" ", experiment_records[i, column_index]) + 1
              pos_2 <- regexpr(" ", substr(experiment_records[i, column_index],
                                           pos_1, 
                                           nchar(experiment_records[i, column_index]))) - 
                2 + pos_1
              experiment_records[i, column_index] <- 
                as.integer(substr(experiment_records[i, column_index], 
                                  pos_1, 
                                  pos_2))
            }
          }
        }
      }
    }
    numeric_cols <- c("pass", "change_threshold", "patience", "seed",
                      "l1_factor", "l2_factor", "learning_rate_1", 
                      "decay_1", "dropout_scheme", 
                      "train_accuracy", "val_accuracy", "test_accuracy",
                      "best_epoch_val", "best_epoch_test", "best_init_epochs",
                      "epochs", "batch_size", "train_val_split")
    for (i in which(colnames(experiment_records) %in% numeric_cols)) {
      experiment_records[, i] <- as.numeric(experiment_records[, i])
    }
    return(experiment_records)
  }
#
#  
#  
#
# end function make_layers_numeric
#
#
# 
#  
  plot_summary_stat <- function(experiment_records,
                                exclude_below_train = 0.5,
                                exclude_below_val = 0.5,
                                exclude_below_test = 0.5) {
#  
# plot results
#
    configs <- unique(experiment_records[, "config"])
#
# look at val accuracy vs. train accuracty to see dependency
#
    bar_bars <- 40
    hist_bars <- 50
    experiment_records_cutoff <- 
      experiment_records[experiment_records[, "val_accuracy"] >= 
                           exclude_below_val &
                           experiment_records[, "train_accuracy"] >= 
                           exclude_below_train, ]
    experiment_records_temp <- 
      experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                  configs[1], ]
    xlim <- 
      c(floor(100 * 
                0.95 * min(experiment_records_temp[, "train_accuracy"], 
                           na.rm = TRUE)) / 100,
        ceiling(100 * 
                  1.05 * max(experiment_records_temp[, "train_accuracy"],
                             na.rm = TRUE)) / 100)
    ylim <- 
      c(floor(100 * 0.95 * exclude_below_val) / 100, 1)
    par(fig = c(0.2, 1, 0.2, 1))
    plot(x = 
           experiment_records_temp[, "train_accuracy"],
         y = 
           experiment_records_temp[, "val_accuracy"],
         col = 1, pch = 1,
         xaxt = "n", xlab = "",
         yaxt = "n", ylab = "",
         xlim = xlim, ylim = ylim,
         main = "All results", 
         cex.main = 1, 
         font.main = 1)
    if (length(configs) > 1) {
      for (i in 2:length(configs)) {
        experiment_records_temp<- 
          experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                      configs[i], ]
        points(x = 
                 experiment_records_temp[, "train_accuracy"],
               y = 
                 experiment_records_temp[, "val_accuracy"],
               col = i, pch = i)
      }
    }
    par(fig = c(0.2, 1, 0, 0.55), new = TRUE)
    hist(experiment_records_temp[, "train_accuracy"],
         breaks = seq(xlim[1], xlim[2], 
                      ((xlim[2] - xlim[1]) / hist_bars)), 
         col = "lightblue",
         yaxt = "n", ylab = "",
         main = "",
         xlab = "train accuracy")
    test_hist <- 
      hist(experiment_records_temp[, "val_accuracy"],
           breaks = seq(ylim[1], ylim[2], 
                        ((ylim[2] - ylim[1]) / bar_bars)),
           plot = FALSE)
    bar_labs <- numeric()
    for (i in 1:(length(test_hist$breaks) - 1)) {
      bar_labs[i] <- 
        floor(100 * mean(test_hist$breaks[i + 1], 
                         test_hist$breaks[i])) / 100
    }
    par(fig = c(0, 0.4, 0.2, 1), new = TRUE)
    barplot(test_hist$counts, 
            col = "lightblue", 
            horiz = TRUE,
            ylab = "", 
            xlab = "", 
            xaxt = "n", 
            names.arg = as.character(bar_labs),
            space = 0)
    mtext("val accuracy", side = 2, line = 3)
    par(fig = c(0, 1, 0, 1))
#
# look at test accuracy vs val accuracy to see generalization
#
    bar_bars <- 40
    hist_bars <- 50
    experiment_records_cutoff <- 
      experiment_records[experiment_records[, "val_accuracy"] >= 
                           exclude_below_val &
                           experiment_records[, "test_accuracy"] >= 
                           exclude_below_test, ]
    experiment_records_temp <- 
      experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                  configs[1], ]
    xlim <- 
      c(floor(100 * 
                0.95 * min(experiment_records_temp[, "val_accuracy"], 
                           na.rm = TRUE)) / 100,
        ceiling(100 * 
                  1.05 * max(experiment_records_temp[, "val_accuracy"],
                             na.rm = TRUE)) / 100)
    ylim <- 
      c(floor(100 * 0.95 * exclude_below_test) / 100, 1)
    par(fig = c(0.2, 1, 0.2, 1))
    plot(x = 
           experiment_records_temp[, "val_accuracy"],
         y = 
           experiment_records_temp[, "test_accuracy"],
         col = 1, pch = 1,
         xaxt = "n", xlab = "",
         yaxt = "n", ylab = "",
         xlim = xlim, ylim = ylim,
         main = "All results", 
         cex.main = 1, 
         font.main = 1)
    if (length(configs) > 1) {
      for (i in 2:length(configs)) {
        experiment_records_temp<- 
          experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                      configs[i], ]
        points(x = 
                 experiment_records_temp[, "val_accuracy"],
               y = 
                 experiment_records_temp[, "test_accuracy"],
               col = i, pch = i)
      }
    }
    par(fig = c(0.2, 1, 0, 0.55), new = TRUE)
    hist(experiment_records_temp[, "val_accuracy"],
         breaks = seq(xlim[1], xlim[2], 
                      ((xlim[2] - xlim[1]) / hist_bars)), 
         col = "lightblue",
         yaxt = "n", ylab = "",
         main = "",
         xlab = "val accuracy")
    test_hist <- 
      hist(experiment_records_temp[, "test_accuracy"],
           breaks = seq(ylim[1], ylim[2], 
                        ((ylim[2] - ylim[1]) / bar_bars)),
           plot = FALSE)
    bar_labs <- numeric()
    for (i in 1:(length(test_hist$breaks) - 1)) {
      bar_labs[i] <- 
        floor(100 * mean(test_hist$breaks[i + 1], 
                         test_hist$breaks[i])) / 100
    }
    par(fig = c(0, 0.4, 0.2, 1), new = TRUE)
    barplot(test_hist$counts, 
            col = "lightblue", 
            horiz = TRUE,
            ylab = "", 
            xlab = "", 
            xaxt = "n", 
            names.arg = as.character(bar_labs),
            space = 0)
    mtext("test accuracy", side = 2, line = 3)
    par(fig = c(0, 1, 0, 1))
#
# look at test f1 vs val f1 to see generalization
#
    bar_bars <- 40
    hist_bars <- 50
    experiment_records_cutoff <- 
      experiment_records[experiment_records[, "val_f1"] >= 
                           exclude_below_val &
                           experiment_records[, "test_f1"] >= 
                           exclude_below_test, ]
    experiment_records_temp <- 
      experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                  configs[1], ]
    xlim <- 
      c(floor(100 * 
                0.95 * min(experiment_records_temp[, "val_f1"], 
                           na.rm = TRUE)) / 100,
        ceiling(100 * 
                  1.05 * max(experiment_records_temp[, "val_f1"],
                             na.rm = TRUE)) / 100)
    ylim <- 
      c(floor(100 * 0.95 * exclude_below_test) / 100, 1)
    par(fig = c(0.2, 1, 0.2, 1))
    plot(x = 
           experiment_records_temp[, "val_f1"],
         y = 
           experiment_records_temp[, "test_f1"],
         col = 1, pch = 1,
         xaxt = "n", xlab = "",
         yaxt = "n", ylab = "",
         xlim = xlim, ylim = ylim,
         main = "All results", 
         cex.main = 1, 
         font.main = 1)
    if (length(configs) > 1) {
      for (i in 2:length(configs)) {
        experiment_records_temp <- 
          experiment_records_cutoff[experiment_records_cutoff[, "config"] ==
                                      configs[i], ]
        points(x = 
                 experiment_records_temp[, "val_f1"],
               y = 
                 experiment_records_temp[, "test_f1"],
               col = i, pch = i)
      }
    }
    par(fig = c(0.2, 1, 0, 0.55), new = TRUE)
    hist(experiment_records_temp[, "val_f1"],
         breaks = seq(xlim[1], xlim[2], 
                      ((xlim[2] - xlim[1]) / hist_bars)), 
         col = "lightblue",
         yaxt = "n", ylab = "",
         main = "",
         xlab = "val f1")
    test_hist <- 
      hist(experiment_records_temp[, "test_f1"],
           breaks = seq(ylim[1], ylim[2], 
                        ((ylim[2] - ylim[1]) / bar_bars)),
           plot = FALSE)
    bar_labs <- numeric()
    for (i in 1:(length(test_hist$breaks) - 1)) {
      bar_labs[i] <- 
        floor(100 * mean(test_hist$breaks[i + 1], 
                         test_hist$breaks[i])) / 100
    }
    par(fig = c(0, 0.4, 0.2, 1), new = TRUE)
    barplot(test_hist$counts, 
            col = "lightblue", 
            horiz = TRUE,
            ylab = "", 
            xlab = "", 
            xaxt = "n", 
            names.arg = as.character(bar_labs),
            space = 0)
    mtext("test f1", side = 2, line = 3)
    par(fig = c(0, 1, 0, 1))
  }
#
#
#
#
# end function plot_summary_stat
#
#
#
#
#
  boxplot_results <- function(experiment_records, 
                              dep_var = "val_accuracy",
                              exclude_below = 0.5,
                              par_list) {
#    
    box_plot_it <- function(dep_var, ind_var, ind_vars, 
                            data_temp, single_OK = FALSE) {
      par(fig = c(0, 1, 0, 1))
      plot_it <- FALSE
      if (single_OK) {
        plot_it <- TRUE
      } else if (!(is.null(nrow(ind_vars)))) {
        if (nrow(ind_vars) > 1) {
          plot_it <- TRUE
        }
      } else if (length(ind_vars) > 1) {
        plot_it <- TRUE
      }
      if (plot_it) {
        main_title <- paste0(gsub(pattern = "_", replacement = " ", dep_var), 
                             " vs. ", 
                             gsub(pattern = "_", replacement = " ", ind_var))
        form <- as.formula(paste0(dep_var, " ~ ", ind_var))
        boxplot(form, 
                data = data_temp,
                main = main_title)
        points(x = as.factor(data_temp[, ind_var]), 
               y = data_temp[, dep_var],
               pch = 20,
               col = "red")
      }
    }
#
#    
#
#
# end function box_plot_it
#
#
#
#
    unlist(par_list)
    data_temp <- 
      experiment_records[experiment_records[, dep_var] >= exclude_below, ]
#
    box_plot_it(dep_var, "layer_14",
                units_used, data_temp,
                single_OK = TRUE)
    box_plot_it(dep_var, "dropout_scheme",
                dropouts, data_temp)
    box_plot_it(dep_var, "train_val_split", 
                train_val_splits, data_temp)
    box_plot_it(dep_var, "batch_size",
                batch_sizes_used, data_temp)
    box_plot_it(dep_var, "learning_rate_1",
                learning_rates, data_temp)
    box_plot_it(dep_var, "l1_factor",
                l1_factors, data_temp)
    box_plot_it(dep_var, "l2_factor",
                l2_factors, data_temp)
    box_plot_it(dep_var, "decay_1",
                decays, data_temp)
  }
#
#  
#  
#
# end function boxplot_results
#
#
#
#
  get_peak <- function(history, smoothing = 3, 
                       direction = "max") {
    history_smooth <-
      rollmean(history[["metrics"]][["val_acc"]],
               smoothing, 
               align = c("center"),
               fill = c("extend",
                        "extend",
                        "extend"))
    if (direction == "max") {
      best_epoch <- max(which(history_smooth ==
                  max(history_smooth)))
    } else {
      best_epoch <- min(which(history_smooth ==
                                min(history_smooth)))
    }
    return(best_epoch)
  }
#
#
#
#
# end function get_peak
#
#
#
#
  get_peak_generic <- function(data, smoothing = 3, 
                       direction = "max") {
    smooth_data <-
      rollmean(data,
               smoothing, 
               align = c("center"),
               fill = c("extend",
                        "extend",
                        "extend"))
    if (direction == "max") {
      peak <- max(which(smooth_data ==
                                max(smooth_data)))
    } else {
      peak <- min(which(smooth_data ==
                                min(smooth_data)))
    }
    return(peak)
  }
#
#
#
#
# end function get_peak_generic
#
#
#
#
  convert_model <- function(model_R) {
    mod_layers <- 
      character(length = length(model_R[["config"]][["layers"]]))
    mod_functions <- 
      character(length = length(model_R[["config"]][["layers"]]))
    mod_params <- 
      character(length = length(model_R[["config"]][["layers"]]))
    layer_types <- 
      character(length  = length(model_R[["config"]][["layers"]]))
    for (i in 1:length(model_R[["config"]][["layers"]])) {
      mod_layers[i] <- model_R[["config"]][["layers"]][[i]][["name"]]
      layer_types[i] <- model_R[["config"]][["layers"]][[i]][["class_name"]]
      if(model_R[["config"]][["layers"]][[i]][["class_name"]] == "Dense") {
        mod_params[i] <- model_R[["config"]][["layers"]][[i]][["config"]][["units"]]
        mod_functions[i] <- model_R[["config"]][["layers"]][[i]][["config"]][["activation"]]
      }
    }
    converted_model <- list(model_R = model_R, 
                       mod_layers = mod_layers, 
                       mod_functions = mod_functions, 
                       mod_params = mod_params, 
                       layer_types = layer_types)
    return(converted_model)
  }
#
#
#
#
# end function convert_model
#
#
#
#
  get_experiment_factors <- function() {
    experiment_factors <- 
      c("date",
        "config",
        "pass",
        "optimizer",
        "change_threshold",
        "patience",
        "stopping_var",
        "seed",
        "l1_factor",
        "l2_factor",
        "learning_rate_1",
        "learning_rate_2",
        "decay_1",
        "decay_2",
        "dropout_scheme",
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "layer_5",
        "layer_6",
        "layer_7",
        "layer_8",
        "layer_9",
        "layer_10",
        "layer_11",
        "layer_12",
        "layer_13",
        "layer_14",
        "layer_15",
        "layer_16",
        "layer_17",
        "layer_18",
        "layer_19",
        "layer_20",
        "layer_21",
        "layer_22",
        "epochs",
        "batch_size",
        "train_val_split",
        "train_accuracy",
        "train_acc_at_best_val",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "best_epoch_val",
        "best_acc_val",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "best_init_epochs")
#    
    return(experiment_factors)
  }
#
#
#
#
# end function get_experiment_factors
#
#
#
#  
  show_summary <- function(par_list, experiment_records) {
    pass <- 0
    unlist(par_list)
    for (trial in 1:replicates) {
      for (unit_index in 1:unit_passes) {
        for (dropout_index in 1:dropout_passes) {
          for(learning_rate_index in 1:length(learning_rates)) {
            for (decay_index in 1:length(decays)) {
              for (epoch_index in 1:length(epochs_used)) {
                for (batch_size_index in 1:length(batch_sizes_used)) {
                  for (l1_factor in l1_factors) {
                    for (l2_factor in l2_factors) {
                      for (train_val_split in train_val_splits) {
                        pass <- pass + 1
                        cat("pass ", pass, " val acc ", 
                            experiment_records[pass, "val_accuracy"],
                            " units ", units_used[unit_index, ],
                            " LR ", learning_rates[learning_rate_index], 
                            " L1 ", l1_factor, 
                            " L2 ", l2_factor, 
                            " batch sz ", batch_sizes_used[batch_size_index], 
                            " dropout ", dropouts[dropout_index, 1], "\n")
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
# 
#
#
#
# end function show_summary
#  
#
#
#  
  init_dropouts <- function() {
    dropout_scheme <- matrix(0, nrow = 20, ncol = 10)
#    
    dropout_scheme[1, ] <- rep(0, 10)
    dropout_scheme[2, ] <- rep(0.25, 10)
#
    return(dropout_scheme)
  }
#  
#
#
# end function init_dropouts
#
#
#  
  init_units <- function() {
    unit_structures <- matrix(0, nrow = 100, ncol = 10)
#    
    unit_structures[1, ] <- c(rep(15, 3), rep(0, 7))
#
    return(unit_structures)
  }
#    
#
#
#
# end function init_units
#
#
#
#
  init_losses <- function() {
    loss_functions <- character(length = 10)
#
    loss_functions[1] <- "mean_absolute_error"
    loss_functions[2] <- "mean_absolute_percentage_error"
    loss_functions[3] <- "squared_hinge"
    loss_functions[4] <- "hinge"
    loss_functions[5] <- "categorical_hinge"
    loss_functions[6] <- "logcosh"
    loss_functions[7] <- "categorical_crossentropy"
    loss_functions[8] <- "kullback_leibler_divergence"
    loss_functions[9] <- "binary_crossentropy"
    loss_functions[10] <- "mean_squared_error"
#
    return(loss_functions)
  }
#
#  
#  
#
# end function init_losses
#
#
#
#
  RMSE_pct <- function(y_true, y_pred) {
    K <- backend()
    y_pred <- K$eval(y_pred)
    y_true <- K$eval(y_true)
    RMSE <- 100 * sqrt(mean((y_pred - y_true)^2)) / mean(y_true)
    RMSE <- K$constant(RMSE)
  }
#
#
#
#
# end function RMSE_pct
#
#
#
#
  learning_rate_function <- function(epoch, lr) {
    how_often <- 3
    if (epoch %% how_often == 0) {
      lr <- lr
    } else {
      lr <- lr + 0.05 * lr
    }
    return(lr)
  }
#
#
#
#
# end function learning_rate_function
#
#
#
#
#
  decode_model <- function(model) {
    coefs <- as.numeric(model[["coefficients"]])
    terms <- as.character(model[["terms"]][[3]])[2:(length(coefs) - 1)]
    terms <- strsplit(terms, " ")
    terms_split <- character()
    for (i in 1:length(terms)) {
      for (j in 1:length(terms[[i]])) {
        terms_split <- c(terms_split, terms[[i]][j])
      }
    }
    terms_split <- terms_split[terms_split != "+"]
    model_list <- list(terms_split, coefs)
    return(model_list)
  }
#
#
#
#
# end fucntion decode_model
#
#
#
#
#
  experiment_records_total <- NULL
#  
  train_data <- read.csv("bitcoin_history.csv",
                         header = TRUE, 
                         skip = 1,
                         stringsAsFactors = FALSE)
#
  train_data %>%
    ggplot(aes(x = as.Date(Date, origin = '1899-12-30'), y = Close)) +
    geom_line(color = "blue") +
    theme(axis.text.x = element_text(size = 12, hjust = 0.5, angle = 90)) +
    theme(axis.title.x = element_text(size = 14)) +
    scale_x_date(name = "", 
                 labels = date_format("%Y-%m-%d")) +
    theme(axis.text.y = element_text(size = 12)) +
    theme(axis.title.y = element_text(size = 14)) +
    scale_y_continuous(name = paste0("Closing Price"),
                       labels = dollar_format()) +
    theme(axis.title.y = element_text(margin = margin(r = 10))) +
    ggtitle(paste0("Historical Bitcoin closing price")) +
    theme(plot.title = element_text(size = 18, hjust = 0.5))
#
# zoom in; it appears we can focus on recent data and 
# predict pretty well
#
  train_data %>%
    filter(as.Date(Date, origin = '1899-12-30') > '2017-12-01') %>%
    ggplot(aes(x = as.Date(Date, origin = '1899-12-30'), y = Close)) +
    geom_line(color = "blue") +
    theme(axis.text.x = element_text(size = 12, hjust = 0.5, angle = 90)) +
    theme(axis.title.x = element_text(size = 14)) +
    scale_x_date(name = "", 
                 labels = date_format("%Y-%m-%d")) +
    theme(axis.text.y = element_text(size = 12)) +
    theme(axis.title.y = element_text(size = 14)) +
    scale_y_continuous(name = paste0("Closing Price"),
                       labels = dollar_format()) +
    theme(axis.title.y = element_text(margin = margin(r = 10))) +
    ggtitle(paste0("Historical Bitcoin closing price")) +
    theme(plot.title = element_text(size = 18, hjust = 0.5))
#
# let's cut off the data at the first "clean" valley, ~ 2018-02-01
#
  start_train <- train_data %>%
    filter(as.Date(Date, origin = '1899-12-30') > '2018-01-01' &
             as.Date(Date, origin = '1899-12-30') < '2018-03-01') %>%
    pull(Close) %>%
    get_peak_generic(smoothing = 3, direction = "min")
  start_train <- which(as.Date(train_data[, "Date"], 
                               origin = '1899-12-30') ==
                                 '2018-01-01') + start_train - 1
  train_data <- train_data[start_train:nrow(train_data), ]
#
# look at autocorrelation
# 
  acf(train_data[, "Close"], type = "correlation", plot = TRUE)
#
# this shows that the closing price is highly self-correlated at
# small lags; thus we can try to predict one week in advance 
# with a simple 7-day lag of close, and keep the volume and date
#
# create lagged feature
#
  close_lag <- 7
  lagged_data <- 
    train_data[1:(nrow(train_data) - (close_lag)), ]
  closing <- train_data[(close_lag + 1):nrow(train_data), "Close"]
  train_data <- cbind(lagged_data, closing)
#  
# select features
#
  keep_features <-
    which(colnames(train_data) %in%
            c("Date", "Close", "Volume", "closing"))
  train_data <- train_data[, keep_features]
#
  train_data_temp <- as.matrix(train_data)
#
# scale data
#
  train_data_temp <- scale(train_data_temp)
#
# set up trials
#
  experiment_factors <- get_experiment_factors()
#
# parameers for dense layers
#
  activations_avail <- c("linear", "hard_sigmoid", 
                         "sigmoid", "softmax",
                         "elu", "selu",
                         "softplus", "softsign",
                         "relu", "tanh")
#  
  activations <- rep(activations_avail[3], 10)
  par_list <- list(activations = activations)
#  
  loss_functions <- init_losses()
  which_losses <- c(10)
  loss_function_used <- loss_functions[which_losses]
  par_list <- c(par_list, loss_function_used = loss_function_used)
#
  unit_structures <- init_units()
  which_units <- c(1)
  units_used <- NULL
#  
  for (i in 1:length(which_units)) {
    units_used <- rbind(units_used,
                        unit_structures[which_units[i], ])
  }
  if (is.null(nrow(units_used))) {
    unit_passes <- 1
  } else {
    unit_passes <- nrow(units_used)
  }
  par_list <- c(par_list, units_used = units_used, 
                unit_passes = unit_passes)
#  
  dropout_scheme <- init_dropouts()
  which_dropouts <- c(1)
  dropouts <- NULL
#  
  for (i in 1:length(which_dropouts)) (
    dropouts <- rbind(dropouts, 
                      dropout_scheme[which_dropouts[i], ])
  )
  if (is.null(nrow(dropouts))) {
    dropout_passes <- 1
  } else {
    dropout_passes <- nrow(dropouts)
  }
  par_list <- c(par_list, dropouts = dropouts, 
                dropout_passes = dropout_passes)
#  
# overall optimization parameters
#
  train_val_splits <- c(0.75)
  par_list <- c(par_list, train_val_splits = train_val_splits)
#
  learning_rates <- c(1, 0.5, 0.1, 0.01, 0.001)
  par_list <- c(par_list, learning_rates = learning_rates)
#  
  decays <- c(0)
  par_list <- c(par_list, decays = decays)
#  
  epochs_used <- c(30)
  par_list <- c(par_list, epochs_used = epochs_used)
#
  l1_factors <- c(0.000)
  l2_factors <- c(0.000)
  par_list <- c(par_list, l1_factors = l1_factors, 
                l2_factors = l2_factors)
#  
  batch_sizes_used <- c(1)
  par_list <- c(par_list, batch_sizes_used = batch_sizes_used)
#  
# experiment loop
#
  time_stamp <- 
    paste0(as.character(Sys.time(), "%Y-%m-%d-%H-%M-%s"))
#
# configure callbacks
#
  callbacks <- NULL
  change_threshold <- 0
  patience <- 3
#
# configure ealy stopping
#
  use_early_stopping <- FALSE
  if (use_early_stopping) {
    stopping_var <- "val_mean_absolute_percentage_error"
    change_threshold <- 0.0000
#
# don't set paitence less than 3 or it can crash peak finding
#
    patience <- 3
    par_list <- c(par_list, stopping_var = stopping_var, 
                  change_threshold = change_threshold, 
                  patience = patience)
    callbacks <- list(callback_early_stopping(monitor = stopping_var,
                                              min_delta = change_threshold,
                                              patience = patience))
  }
#
# configure tensorboard
#
  use_tensorboard <- FALSE
  if (use_tensorboard) {
    dir_run <- paste0("my_logs_", time_stamp)
    dir.create(dir_run)
    tensorboard(dir_run)
    callbacks <- c(callbacks,
                   callback_tensorboard(log_dir = dir_run,
                                        histogram_freq = 100))
  }
#
# use custom learning rate noise function
#
  use_lr_function <- FALSE
  if (use_lr_function) {
    callbacks <- c(callbacks,
                   callback_learning_rate_scheduler(learning_rate_function))
  }
#
# determine if plots go to pdf
#
  pdf_plots <- FALSE
  if (pdf_plots) {
    pdf(file = paste0(time_stamp, "_crypto_nn_charts.pdf"), 
        onefile = TRUE)
  }
#  
  experiment_records <- 
    as.data.frame(matrix(nrow = 1,
                         ncol = length(experiment_factors)))
  colnames(experiment_records) <- experiment_factors
#  
# initialize for convenience in testing
#
  trial <- 1
  unit_index <- 1
  dropout_index <- 1
  learning_rate_index <- 1
  decay_index <- 1
  epoch_index <- 1
  batch_size_index <- 1
  l1_factor <- l1_factors[1]
  l2_factor <- l2_factors[2]
  train_val_split <- 0.75
  experiment_record <- 
    as.data.frame(matrix(nrow = 1,
                         ncol = length(experiment_factors)))
  colnames(experiment_record) <- experiment_factors
#
# initialize/configure for run
#
  replicates <- 1
  pass <- 0
  prior_pass <- 0
  decode_current_model <- TRUE
  post_plots <- TRUE
  boxplots <- TRUE
  save_results_fine <- TRUE
  save_results_summary <- TRUE
  optimizer <- "RMSprop"
  run_date <- as.character(Sys.time(), "%Y-%m-%d")
#
  configuration <- 
    paste0("initial trial using simple lagged data", 
           paste0(unique(activations), " "), "activations")
#
# construct a test set
#
  test_split <- 0.10
  test_indices <- sample(seq(1, nrow(train_data_temp), 1),
                         test_split * nrow(train_data_temp),
                         replace = FALSE)
  test_data <- train_data_temp[test_indices, ]
  train_data_temp <- train_data_temp[- test_indices, ]
  for (trial in 1:replicates) {
    for (unit_index in 1:unit_passes) {
      for (dropout_index in 1:dropout_passes) {
        for(learning_rate_index in 1:length(learning_rates)) {
          for (decay_index in 1:length(decays)) {
            for (epoch_index in 1:length(epochs_used)) {
              for (batch_size_index in 1:length(batch_sizes_used)) {
                experiment_record <- 
                  as.data.frame(matrix(nrow = 1,
                                       ncol = length(experiment_factors)))
                colnames(experiment_record) <- experiment_factors
                for (l1_factor in l1_factors) {
                  for (l2_factor in l2_factors) {
                    for (train_val_split in train_val_splits) {
                      cat("\014")
                      seed <- as.integer(100 * runif(1, min = 0, max = 10000))
                      use_session_with_seed(seed, disable_gpu = FALSE, 
                                            disable_parallel_cpu = FALSE)
#
# split train into train and val and remove target columns
#
                      target_col <- which(colnames(train_data_temp) == "closing")
                      train_indices <- 
                        sample(1:nrow(train_data_temp),
                               train_val_split * nrow(train_data_temp))
                      val_data <- train_data_temp[- train_indices, - target_col]
                      train_data <- 
                        train_data_temp[train_indices, - target_col]
                      pass <- pass + 1
#  
# we have to reassign these variables because during plotting we
# change some of them to data.frames so we need to convert
# back to matrix form
#
                      x_train <- 
                        as.matrix(train_data)
                      x_test <- 
                        as.matrix(test_data[, -target_col])
                      x_val <- 
                        as.matrix(val_data)
                      y_train <- 
                        as.matrix(train_data_temp[train_indices, target_col])
                      y_test <- 
                        as.matrix(test_data[, target_col])
                      y_val <- 
                        as.matrix(train_data_temp[- train_indices, target_col])
#
                      input_data <- list(x_train)      
#                      
                      numerical_in <- 
                        layer_input(shape = ncol(x_train),
                                    name = "input_numerical")
#              
# combine the input layers (note: for future use if we add categoriical data)
# 
                      combined_inputs <-
                        numerical_in
#                        layer_concatenate(c(numerical_in))
#
# define first layer
#                      
                      combined_model <- combined_inputs %>%
                        layer_dense(units = (ncol(x_train)), 
                        activation = activations[1]) %>%
                        layer_dropout(rate = 
                                        if(dropout_passes == 1) {
                                          dropouts[1] 
                                        } else {
                                          dropouts[dropout_index, 1]}) %>%
                        layer_dense(units = 
                                      if (unit_passes == 1) {
                                        units_used[1]
                                      } else {
                                        units_used[unit_index, 1]}, 
                                    activation = activations[2],
                                    kernel_regularizer = 
                                      regularizer_l1_l2(l1 = l1_factor, l2 = l2_factor))
#
# define rest of layers
# 
                      if (unit_passes == 1) {
                        temp_units <- units_used
                      } else {
                        temp_units <- units_used[unit_index, ]
                      }
                      for (i in 2:length(temp_units)) {
                        if (temp_units[i] > 0) {
                          combined_model <- combined_model %>%
                            layer_dropout(rate = 
                                            if(dropout_passes == 1) {
                                              dropouts[2] 
                                            } else {
                                              dropouts[dropout_index, 2]}) %>%
                            layer_dense(units = 
                                          if (unit_passes == 1) {
                                            units_used[2]
                                          } else {
                                            units_used[unit_index, 2]}, 
                                        activation = activations[3],
                                        kernel_regularizer = 
                                          regularizer_l1_l2(l1 = l1_factor, l2 = l2_factor))
                        }
                      }
#
# define output layer
#
                      combined_model <- combined_model %>%
                        layer_dense(units = 1, activation = "linear")
#    
                      model <- 
                        keras::keras_model(inputs = c(numerical_in), 
                                           outputs = combined_model)
                      optimizer_string <- paste0("optimizer = optimizer_",
                                                 optimizer, "(lr = ",
                                                 learning_rates[learning_rate_index], 
                                                 ", decay = ",
                                                 decays[decay_index], ")")
                      if (optimizer == "sgd") {
                        optimizer_used <- 
                          optimizer_sgd(lr = learning_rates[learning_rate_index],
                                        decay = decays[decay_index])
                      } else {
                        optimizer_used <- 
                          optimizer_rmsprop(lr = learning_rates[learning_rate_index])
                      }
                      model %>% compile(
                        optimizer = 
                          optimizer_used,
                        loss = loss_function_used,
                        metrics = "mean_absolute_percentage_error"
                      )
#  
                      history <- model %>% fit(
                        x = input_data,
                        y = y_train,
                        shuffle = FALSE,
                        view_metrics = FALSE,
                        epochs = epochs_used[epoch_index],
                        batch_size = batch_sizes_used[batch_size_index],
                        validation_data = 
                          list(list(x_val), 
                               y_val),
                        verbose = 1,
                        callbacks = callbacks
                      )
#
                      val_predictions <- 
                        predict(model, list(x_val))
                      val_metrics <- get_metrics(val_predictions, y_val)
                      val_predictions <- val_metrics[[1]]
                      val_accuracy <- val_metrics[[2]]
                      val_precision <- val_metrics[[3]]
                      val_recall <- val_metrics[[4]]
                      val_f1 <- val_metrics[[5]]
#
                      test_predictions <- 
                        predict(model, list(x_test))
                      test_metrics <- get_metrics(test_predictions, y_test)
                      test_predictions <- test_metrics[[1]]
                      test_accuracy <- test_metrics[[2]]
                      test_precision <- test_metrics[[3]]
                      test_recall <- test_metrics[[4]]
                      test_f1 <- test_metrics[[5]]
#
# save the predictions for the test data
#
                      if (pass == 1) {
                        test_list <- 
                          list(cbind(test_predictions, y_test))
                      } else {
                        test_list <- 
                          c(test_list, list(cbind(test_predictions, y_test)))
                      }
# 
                      x_train <- as.data.frame(x_train)
                      x_val <- as.data.frame(x_val)
                      x_test <- as.data.frame(x_test)
#
# extract model to json
#
                      model_R <- fromJSON(model_to_json(model))
                      converted_model <- convert_model(model_R)
                      mod_layers <- converted_model[[2]]
                      mod_functions <- converted_model[[3]]
                      mod_params <- converted_model[[4]]
                      layer_types <- converted_model[[5]]
#
# find best epoch (so we can use later for retraining to optimum)
#                    
                      best_init_epochs <- get_peak(history, 
                                                   smoothing = 3,
                                                   direction = "max")
                      best_epoch_val <- best_init_epochs
#                      
                      max_epoch <- length(history[["metrics"]][["acc"]])
#
                      experiment_record[pass - prior_pass, ]$date <-
                        run_date
                      experiment_record[pass - prior_pass, ]$config <-
                        configuration
                      experiment_record[pass - prior_pass, ]$pass <-
                        pass
                      experiment_record[pass - prior_pass, ]$optimizer <-
                        optimizer
                      experiment_record[pass - prior_pass, ]$change_threshold <-
                        change_threshold
                      experiment_record[pass - prior_pass, ]$patience <-
                        patience
                      experiment_record[pass - prior_pass, ]$stopping_var <-
                        stopping_var
                      experiment_record[pass - prior_pass, ]$seed <-
                        seed
                      experiment_record[pass - prior_pass, ]$l1_factor <-
                        l1_factor
                      experiment_record[pass - prior_pass, ]$l2_factor <-
                        l2_factor
                      experiment_record[pass - prior_pass, ]$learning_rate_1 <-
                        learning_rates[learning_rate_index]
                      experiment_record[pass - prior_pass, ]$decay_1 <-
                        decays[decay_index]
                      experiment_record[pass - prior_pass, ]$dropout_scheme <-
                        which_dropouts[dropout_index]
                      first_layer_col <- 
                        which(colnames(experiment_record) == "layer_1")
                      total_layers <- length(mod_layers)
                      for (i in first_layer_col:(first_layer_col + total_layers - 1)) {
                        experiment_record[pass - prior_pass, i] <-
                          paste0(mod_layers[i - first_layer_col + 1], " ",
                                 mod_params[i - first_layer_col + 1], " ",
                                 mod_functions[i - first_layer_col + 1])
                      }
                      experiment_record[pass - prior_pass, ]$epochs <-
                        paste0(epochs_used[epoch_index])
                      experiment_record[pass - prior_pass, ]$batch_size <-
                        batch_sizes_used[batch_size_index]
                      experiment_record[pass - prior_pass, ]$train_val_split <-
                        train_val_split
                      experiment_record[pass - prior_pass, ]$train_accuracy <-
                        history[["metrics"]][["acc"]][max_epoch]
                      experiment_record[pass - prior_pass, ]$train_acc_at_best_val <-
                        history[["metrics"]][["acc"]][best_epoch_val]
                      experiment_record[pass - prior_pass, ]$val_accuracy <-
                        history[["metrics"]][["val_acc"]][max_epoch]
                      experiment_record[pass - prior_pass, ]$val_precision <-
                        val_precision
                      experiment_record[pass - prior_pass, ]$val_recall <-
                        val_recall
                      experiment_record[pass - prior_pass, ]$val_f1 <-
                        val_f1
                      experiment_record[pass - prior_pass, ]$best_acc_val <-
                        history[["metrics"]][["val_acc"]][best_epoch_val]
                      experiment_record[pass - prior_pass, ]$best_epoch_val <-
                        best_epoch_val
                      experiment_record[pass - prior_pass, ]$test_accuracy <-
                        test_accuracy
                      experiment_record[pass - prior_pass, ]$test_precision <-
                        test_precision
                      experiment_record[pass - prior_pass, ]$test_recall <-
                        test_recall
                      experiment_record[pass - prior_pass, ]$test_f1 <-
                        test_f1
                      experiment_record[pass - prior_pass, ]$best_init_epochs <-
                        best_init_epochs
                      plot(1, 1, type = "l", axes = FALSE, 
                           cex.lab = 0.01, 
                           main = c(rep("\n", length(experiment_factors) / 2),
                                    paste("Val Data\n\n", 
                                          paste(experiment_factors,
                                                experiment_record[pass - prior_pass, ], 
                                                collapse = "\n"))),
                           cex.main = 0.5,
                           font.main = 1)
                      par(mar = c(5, 4, 4, 4) + 0.1)
                      ylim <- c(max(0, 0.9 * min(min(history[["metrics"]][["loss"]]),
                                                 min(history[["metrics"]][["val_loss"]]))),
                                1.1 * max(max(history[["metrics"]][["loss"]]),
                                          max(history[["metrics"]][["val_loss"]])))
                      ylim <- c(floor(10 * ylim[1]) / 10,
                                round(10 * ylim[2] / 10, 1))
                      plot(history[["metrics"]][["val_loss"]],
                           type = "l",
                           col = "blue",
                           lwd = 0.5,
                           yaxt = "n",
                           ylab = "",
                           ylim = ylim,
                           xaxt = "n",
                           xlab = "epoch")
                      axis(side = 1, 
                           at = seq(1, 
                                    length(history[["metrics"]][["val_loss"]]), 
                                    1))
                      axis(side = 4)
                      lines(history[["metrics"]][["loss"]],
                            type = "l",
                            col = "darkgreen",
                            lwd = 0.5)
                      par(new = T)
                      ylim <- c(max(0, 0.9 * min(min(history[["metrics"]][["acc"]]),
                                                 min(history[["metrics"]][["val_acc"]]))),
                                1.1 * max(max(history[["metrics"]][["acc"]]),
                                          max(history[["metrics"]][["val_acc"]])))
                      ylim <- c(floor(10 * ylim[1]) / 10,
                                round(10 * ylim[2] / 10, 1))
                      plot(history[["metrics"]][["val_acc"]],
                           type = "l",
                           col = "red",
                           lwd = 0.5,
                           ylim = ylim,
                           ylab = "",
                           xaxt = "n",
                           xlab = "")
                      grid(lty = 1, lwd = 0.5)
                      lines(c(history[["metrics"]][["acc"]]),
                            col = "purple",
                            lwd = 0.5)
                      mtext(side = 4, 
                            line = 2.5, 
                            at = 0.5 - 0.08, 
                            "val", col = "blue")
                      mtext(side = 4, 
                            line = 2.5, 
                            at = 0.5 + 0.07,
                            "     / train loss (mse)", 
                            col = "darkgreen")
                      mtext(side = 2, 
                            line = 2.5, 
                            at = 0.5 - 0.07,
                            "train", 
                            col = "purple")
                      mtext(side = 2, 
                            line = 2.5,
                            at = 0.5 + 0.07,
                            "     / val accuracy", 
                            col = "red")
#
# store the models
# 
                      if (pass == 1) {
                        model_list <- list(model)
                      } else {
                        model_list <- c(model_list, list(model))
                      }
                    }
                  }
                }
                if (save_results_fine) {
                  time_stamp <- 
                    as.character(Sys.time(), "%Y-%m-%d-%H-%M-%s")
                  write.csv(experiment_record, paste0(time_stamp, 
                                                      "_nn_experiment_record.csv"))
                }
                if (prior_pass == 0) {
                  experiment_records <- experiment_record
                } else {
                  experiment_records <- rbind(experiment_records,
                                              experiment_record)
                }
                prior_pass <- pass
              }
            }
          }
        }
      }
    }
  }
#  
  show_summary(par_list, experiment_records)
#
  if (save_results_summary) {
    time_stamp <- 
      paste0(as.character(Sys.time(), "%Y-%m-%d-%H-%M-%s"))
    write.csv(experiment_records, paste0(time_stamp,
                                         "_nn_experiments_summary.csv"))
    write.csv(as.data.frame(submission_list), paste0(time_stamp,
                                      "_nn_predictions_summary.csv"))
  }
#
# convert layer text to numeric for charting
#
  make_layers_numeric(experiment_records, mod_layers)
#
  train_cutoff <- 0.5
  val_cutoff <- 0.5
  test_cutoff <- 0.5
#  
  if (post_plots) {
    plot_summary_stat(experiment_records, 
                      train_cutoff, val_cutoff, test_cutoff)
  }
#
  if (boxplots) {
    boxplot_results(experiment_records, 
                    dep_var = "val_accuracy", 
                    exclude_below = val_cutoff, 
                    par_list)
#
    boxplot_results(experiment_records, 
                    dep_var = "test_accuracy", 
                    exclude_below = test_cutoff, 
                    par_list)
    boxplot_results(experiment_records, 
                    dep_var = "test_f1", 
                    exclude_below = test_cutoff, 
                    par_list)
  }
  if (pdf_plots) {
    while (!is.null(dev.list())) {dev.off()}
  }
  if (is.null(experiment_records_total)) {
    experiment_records_total <- experiment_records
  } else {
    experiment_records_total <- rbind(experiment_records_total,
                                      experiment_records)
  }
#
  show_summary(par_list = par_list, experiment_records = experiment_records)