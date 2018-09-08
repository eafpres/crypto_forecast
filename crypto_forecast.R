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
# forecast gets correlation analysis etc.
#
  library(forecast)
#
# scales provides some convenient functions for plotting
#
#
  get_metrics <- function(predictions, labels) {
    error <- (labels - predictions)
    abs_error <- abs(error)
    MAE <- mean(abs_error)
    metrics <- list(MAE = MAE)
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
                      "decay_1", "dropout_scheme","which_units",  "epochs", 
                      "train_MAE", "train_MAE_at_best_val", "val_MAE", 
                      "best_epoch_val", "best_MAE_val", "test_MAE",
                      "batch_size", "train_val_split")
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
  boxplot_results <- function(experiment_records, 
                              dep_var = "val_MAE",
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
    box_plot_it(dep_var, "which_units",
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
  get_peak <- function(history, metric, smoothing = 3, 
                       direction = "max") {
    history_smooth <-
      rollmean(history[["metrics"]][[metric]],
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
        "decay_1",
        "dropout_scheme",
        "which_units",
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
        "train_MAE",
        "train_MAE_at_best_val",
        "val_MAE",
        "best_epoch_val",
        "best_MAE_val",
        "test_MAE")
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
                        cat("pass ", pass, " val MAE ", 
                            experiment_records[pass, "val_MAE"],
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
    dropout_scheme[3, ] <- rep(0.5, 10)
    dropout_scheme[4, ] <- rep(0.1, 10)
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
    unit_structures[2, ] <- c(5, 3, rep(0, 8))
    unit_structures[3, ] <- c(rep(25, 4), rep(0, 6))
    unit_structures[4, ] <- c(rep(35, 4), rep(0, 6))
    unit_structures[5, ] <- c(5, rep(0, 9))
    unit_structures[6, ] <- c(rep(15, 4), rep(0, 6))
    unit_structures[7, ] <- c(rep(15, 5), rep(0, 5))
    unit_structures[8, ] <- c(5, rep(0, 9))
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
    loss_functions[2] <- "mean_absolute_error"
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
  learning_rate_function <- function(epoch, lr) {
    how_often <- 3
    if (epoch %% how_often == 0) {
      lr <- lr + 0.05 * lr
    } else {
      lr <- lr
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
# The goal of this analysis is to predict bitcon
# closing price 7 days in advance
# The appraoch we will take is two-fold
# 1) Use another data set, in this case the nasdaq composite,
#    and try to correlate lagged data to the bitcoin data
# 2) Model the self correlation, trend, and periodic behavior of
#    the bitcon data directly
# The two parts will be combined into a neural network
# model which acts as a non-linear regression model
#
# Note that there are more complex ways to approach
# this problem if more resources are available
# Some include:
# Use bitcoin data from multiple exchanges in different time zones
# Use social media data to model some large movements
# Find additioanl exogenous features that can correlate at
# useful lag times, such as correlating multiple crypto currencies
# and finding one or more that lead bicoin by enough time
# to be useful
#
  nasdaq <- read.csv("nasdaq.csv",
                     header = TRUE,
                     stringsAsFactors = FALSE)
  
  nasdaq <- nasdaq[order(nasdaq[, "Date"], decreasing = FALSE), ]
  nasdaq <- nasdaq %>%
    mutate(Date = as.Date(nasdaq[, "Date"], 
                          origin = '1899-12-30')) 
#
# the nasdaq data omit weekends and holidays, wo
# we'll just impute those so we have uniform data
#
  all_dates <- seq(min(nasdaq[, "Date"]), max(nasdaq[, "Date"]), 1)
  interp_nasdaq <- 
    as.data.frame(matrix(rep(NA, length(all_dates) * ncol(nasdaq)),
                         nrow = length(all_dates), 
                         ncol = ncol(nasdaq)))
  colnames(interp_nasdaq) <- colnames(nasdaq)
  have_data <- which(all_dates %in% nasdaq[, "Date"])
  interp_nasdaq[, "Date"] <- all_dates
  for (i in 2:ncol(interp_nasdaq)) {
    interp_nasdaq[have_data, i] <- nasdaq[, i]
    interp_nasdaq[, i] = na.approx(interp_nasdaq[, i])

  }
  nasdaq <- interp_nasdaq  %>%
    filter(Date >= '2018-01-01')
  nasdaq %>%
    ggplot(aes(x = Date, y = Close)) +
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
    ggtitle(paste0("Historical nasdaq composite closing price")) +
    theme(plot.title = element_text(size = 18, hjust = 0.5))
  nasdaq %>%
    select(Close) %>%
    Acf(lag.max = 60, 
        plot = TRUE, 
        main = "nasdaq")
  nasdaq %>%
    select(Close) %>%
    Pacf(lag.max = 60, 
         plot = TRUE, 
         ylim = c(-0.2, 1), 
         main = "nasdaq")
  nasdaq %>%
    select(Volume, Close) %>%
    Pacf(lag.max = 60, 
         plot = TRUE)
#
# From this we see that, as expected, the closing price
# is highly self-correlated at low lags, which is not 
# useful in this analysis, however from the Pacf we 
# see there is a correlation at 58 days, so we can try to use
# that to further analyze
# it is also evident there are correlations between volume
# and closing price, so we may be able to leverage that
#  
  bitcoin_data <- read.csv("bitcoin_history.csv",
                         header = TRUE, 
                         skip = 1,
                         stringsAsFactors = FALSE)
  bitcoin_original <- bitcoin_data
  bitcoin_data <- bitcoin_data %>%
    mutate(Date = as.Date(bitcoin_data[, "Date"], 
                                  origin = '1899-12-30')) %>%
    filter(Date >= '2018-01-01')
  bitcoin_data %>%
    ggplot(aes(x = Date, y = Close)) +
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
  bitcoin_data %>%
    select(Close) %>%
    Acf(lag.max = 60, 
        plot = TRUE, 
        main = "bitcoin data", 
        ylim = c(-0.2, 1))
  bitcoin_data %>%
    select(Close) %>%
    Acf(lag.max = 60, 
        plot = TRUE, 
        main = "bitcoin data", 
        ylim = c(-0.2, 1), 
        xlim = c(40, 60))
  bitcoin_data %>%
    select(Close) %>%
    Pacf(lag.max = 60, 
         plot = TRUE, 
         ylim = c(-0.2, 1), 
         main = "bitcoin data")
  Ccf(bitcoin_data[, "Volume"], 
      bitcoin_data[, "Close"], 
      lag.max = 60, 
      plot = TRUE,
      main = "bitcoin data")
  bitcoin_data %>%
    select(Volume, Close) %>%
    Pacf(lag.max = 60,
         plot = TRUE)
#
# From this we see that, as exepcted, the closing price
# is highly self-correlated at low lags, which is not 
# useful in this analysis.
# In this case, there does appear to be a self-correlation at 
# 45 days, which we can attempt to use to create lagged
# bitcoining data
# From the Pacf we see taht there is a significant 
# negative correlation at 20 days, which we might 
# interprete as a rebound effect of some sort, so we can 
# include a 20-day lagged feature in the training data
# From the CCF for Volume & Close, we see the maximum
# correlation  at 0 lag but correlations at bot +/- 45 days
# so we could use lagged Volume as another predictor
# 
# It appears we should focus on recent data 
# given the huge spike that is competely different from
# recent behavior.  We'll try working with only 2018 data.
#
  start_date <- as.Date('2018-01-01')
  bitcoin_data <- bitcoin_data %>%
    filter(Date >= start_date)
#  
# here we shift the start of nasdaq to the same as bitcoin
# then truncate to an even multiple of the main frequency
# in order to not have the time series decomp wrap around
#
  nasdaq_acf <- 58
  nasdaq <- nasdaq %>%
    filter(Date >= start_date)
  nasdaq <- nasdaq[1:(nrow(nasdaq) - (nrow(nasdaq) %% nasdaq_acf)), ]
  first_day <- 1
  first_block <- 1
  last_day <- (nrow(nasdaq) - (first_day - 1)) %% nasdaq_acf
  last_block <- (nrow(nasdaq) + (nasdaq_acf - last_day)) / nasdaq_acf
#  
  nasdaq_ts <- nasdaq %>%
    select(Close) %>%
    ts(start = c(first_block, first_day), 
       end = c(last_block, last_day), frequency = nasdaq_acf)
  nasdaq_time_analysis <- decompose(nasdaq_ts, type = "additive")
  plot(nasdaq_time_analysis)
  na.omit(as.numeric(nasdaq_time_analysis[["trend"]])) %>%
    Acf(type = "correlation", lag.max = 180, plot = TRUE,
        main = "nasdaq trend")
  na.omit(as.numeric(nasdaq_time_analysis[["seasonal"]])) %>%
    Acf(type = "correlation", lag.max = 180, plot = TRUE,
        main = "nasdaq seasonal")
#
# Summarizing what we know about the nasdaq data,
# it is self-correlated at 58 days, using that we can extract
# a seasonal component that accounts for about 350 points
# of varition in the 58 day cycle, which is not bad
# We also know we could correlate the closing to Volume
# at a number of periods if that were useful generating
# predictions of the closing to use as a predictor for 
# the training data
#  
# perform similar analysis on bitcoin using its acf self-correlation
#
  bitcoin_acf <- 45
  bitcoin_data <- bitcoin_data[1:(nrow(bitcoin_data) - 
                                (nrow(bitcoin_data) %% bitcoin_acf)), ]
  first_day <- 1
  first_block <- 1
  last_day <- (nrow(bitcoin_data) - (first_day - 1)) %% bitcoin_acf
  last_block <- (nrow(bitcoin_data) + (bitcoin_acf - last_day)) / bitcoin_acf
  bitcoin_ts <- bitcoin_data %>%
    select(Close) %>%
    ts(start = c(first_block, first_day), 
       end = c(last_block, last_day), frequency = bitcoin_acf)
  bitcoin_time_analysis <- decompose(bitcoin_ts, type = "additive")
  plot(bitcoin_time_analysis)
  na.omit(as.numeric(bitcoin_time_analysis[["trend"]])) %>%
    Acf(type = "correlation", lag.max = 180, plot = TRUE, 
        main = "bitcoin data trend")
  na.omit(as.numeric(bitcoin_time_analysis[["seasonal"]])) %>%
    Acf(type = "correlation", lag.max = 180, plot = TRUE,
        main = "bitcoin data seasonal")
#
# Summarizing what we know about the bitcoin data,
# It is self correlated at 45 days and using that we
# can extract a seasonal component that accounts for 1000 points
# of variation in the 45 day period.  For most recent data,
# that is very significant, so we should be able to use that
# as a predictor
# The trend, is somewhat of an expnential decay in the recent
# period.  There is negative self-correlation in the trend
# at ~ 135 days which we could try to leverage, but that
# may not be that useful
#
# So we can begin by modeling the seasonal component using
# fitted sine/cosines, and include that as a predictor,
# as well as the self-correlated data at a 45 day lag period,
# and an exponential fit to predict the trend
#
# finally, look at cross correlation of nasdaq and bitcoin 
# to understand correlation directly between the 2 series
# to see if we can leverage the nasdaq data
#
# we cut off the negative lags as these represent
# where the bitcoin data in the past correlats to nasdaq
# in the future which is not useful to us here
#
  Ccf(bitcoin_data[, "Close"], nasdaq[, "Close"],
      plot = TRUE, 
      lag.max = 45,
      xlim = c(0, 45),
      main = "bitcoin Close vs. nasdaq Close")
  Ccf(bitcoin_data[, "Close"], nasdaq[, "Volume"],
      plot = TRUE,
      lag.max = 45,
      xlim = c(0, 45),
      main = "bitcoin Close vs. nasdaq Volume")
  Ccf(bitcoin_data[, "Volume"], nasdaq[, "Volume"],
      plot = TRUE,
      lag.max = 45,
      xlim = c(0, 45),
      main = "bitcoin Volume vs. nasdaq Volume")
#
# from this analysis we can draw some conclusions
#
# 1. Both the nasdaq data and the bitcoin data for the 
#    recent period show a clear periodic cycle, but
#    the detailed patterns are different and out of phase.
#    We will include a model of the bitcoin seasonality
#    as a predictor.
# 2. There is a strong negative correlation between nasdaq
#    close and bitcoin close, at low lags.  We can either
#    attempt to use 7 day lag to predict the 1 week ahead
#    for bitcoin, or we could leverage the self-correlation
#    of the nasdaq close to predict future nasdaq, then use that
#    at the optimal lag of 5 days.  We will start with the 
#    simpler case of lagging nasdaq close 7 days
# 3. There is a nearly significant correlation between
#    bitcoin close and nasdaq volume at 32 days which we
#    can include in the model.  We will include this initially.
# 4. The trends for the two series are completely different
#    therefore we likely cannot use the nasdaq trend to
#    predict the long term trend of bitcoin, but it appears
#    we could model the bitcon trend as an exponential decay
#    in the short term, and use that as a predictor.
#  
# build time-based features for model
#
# build a sine/cosine series model for the bitcon seasonality
# 
  day_series <- seq(1, length(bitcoin_time_analysis[["seasonal"]]), 1)
  sin_45 <- sin(2 * pi * day_series / 45)
  cos_45 <- cos(2 * pi * day_series / 45)
  sin_30 <- sin(2 * pi * day_series / 30)
  cos_30 <- sin(2 * pi * day_series / 30)
  sin_22.5 <- sin(2 * pi * day_series / 22.5)
  cos_22.5 <- cos(2 * pi * day_series / 22.5)
  sin_15 <- sin(2 * pi * day_series / 15)
  cos_15 <- cos(2 * pi * day_series / 15)
  sin_11.25 <- sin(2 * pi * day_series / 11.25)
  cos_11.25 <- cos(2 * pi * day_series / 11.25)
  bitcoin_seasonal <- bitcoin_time_analysis[["seasonal"]]
  bitcoin_seasonal <- as.data.frame(cbind(day_series,
                                        bitcoin_seasonal,
                                        sin_45,
                                        cos_45, 
                                        sin_30,
                                        cos_30,
                                        sin_22.5,
                                        cos_22.5,
                                        sin_15,
                                        cos_15,
                                        sin_11.25,
                                        cos_11.25))
  bitcoin_seasonal_model <- lm(bitcoin_seasonal ~ . - day_series, 
                             data = bitcoin_seasonal)
  bitcoin_seasonal <- cbind(bitcoin_seasonal,
                          pred = predict(bitcoin_seasonal_model))
  bitcoin_seasonal %>%
    ggplot(aes(x = bitcoin_seasonal, y = pred, group = 1)) +
    geom_point(color = "red") + 
    ggtitle("goodness of fit bitcoin seasonal pred vs. actual") +
    theme(plot.title = element_text(hjust = 0.5))
  bitcoin_seasonal %>%
    ggplot(aes(x = day_series, y = bitcoin_seasonal, group = 1)) +
    geom_line(color = "black") +
    geom_line(aes(x = day_series, y = pred), color = "red") +
    ggtitle("bitcoin seasonal & sin/cos fit") +
    theme(plot.title = element_text(hjust = 0.5))
#
# This shows we can adequately model the bitcoin seasonality 
# with 5 sets of sines/cosines.  The benefit of this approach 
# is that the model will be self-retraining in the future
# instead of hard-coding the predicted seasonality, we include
# the sine/cosine features and the model will refit them
#
# Construct 7-day lagged nasdaq close data
#
  nasdaq_close_lag_7 <- interp_nasdaq
  nasdaq_close_lag_7 <- nasdaq_close_lag_7 %>%
    mutate(Date = Date + 7) %>%
    filter(Date >= '2018-01-01') %>%
    select(Date, Close)
#
# Construct 32 day lagged nasdaq volume data
#
  nasdaq_vol_lag_32 <- interp_nasdaq
  nasdaq_vol_lag_32 <- nasdaq_vol_lag_32 %>%
    mutate(Date = Date + 32) %>%
    filter(Date >= '2018-01-01') %>%
    select(Date, Volume)
#
# Construct an exponential decay model for the bitcon trend
#
  day_series <- seq(as.Date('2018-01-01'), 
                    as.Date('2018-01-01') + 
                      length(bitcoin_time_analysis[["trend"]]) - 1, 
                    1)
  bitcoin_trend <- 
    as.data.frame(cbind(day_series, trend = bitcoin_time_analysis[["trend"]]))
  exp_model <- lm(log(trend) ~ day_series, data = bitcoin_trend, 
                  na.action = na.exclude)
  A <- exp(exp_model[["coefficients"]][["(Intercept)"]])
  B <- exp_model[["coefficients"]][["day_series"]]
#
# trend = A * exp(B * day_series)
#
# extend to most recent date in bitcoin data
#
  day_series <- seq(as.Date('2018-01-01'), 
                    max(as.Date(bitcoin_original[, "Date"], 
                                origin = '1899-12-30')), 
                    1)
  bitcoin_trend_fit <- 
    as.data.frame(cbind(Date = day_series, 
                        trend_fit = A * exp(B * as.integer(day_series))))
  bitcoin_trend %>%
    ggplot(aes(x = as.Date(day_series, origin = '1899-12-30'), y = trend, group = 1)) +
    geom_point(color = "red") +
    geom_line(data = bitcoin_trend_fit, 
              aes(x = as.Date(Date, origin = '1899-12-30'), y = trend_fit), 
              color = "blue") +
    ggtitle("bitcoin trend and exponential fit") +
    theme(plot.title = element_text(hjust = 0.5))
#
# assemble the training data
#
  forecast_interval <- 7
  train_data <- bitcoin_data %>%
    filter(as.Date(Date, origin = '1899-12-30') >= '2018-01-01') %>%
    select(Date, Close) 
  extend_train <- data.frame(Date = rep(NA, forecast_interval),
                             Close = rep(NA, forecast_interval))
  train_data <- rbind(train_data, extend_train)
  day_series <- seq(1, nrow(train_data), 1)
  sin_45 <- sin(2 * pi * day_series / 45)
  cos_45 <- cos(2 * pi * day_series / 45)
  sin_30 <- sin(2 * pi * day_series / 30)
  cos_30 <- sin(2 * pi * day_series / 30)
  sin_22.5 <- sin(2 * pi * day_series / 22.5)
  cos_22.5 <- cos(2 * pi * day_series / 22.5)
  sin_15 <- sin(2 * pi * day_series / 15)
  cos_15 <- cos(2 * pi * day_series / 15)
  sin_11.25 <- sin(2 * pi * day_series / 11.25)
  cos_11.25 <- cos(2 * pi * day_series / 11.25)
  train_data <- cbind(Date = train_data[, "Date"],
                      bc_trend = bitcoin_trend_fit[1:nrow(train_data), "trend_fit"],
                      nd_c_l_7 = nasdaq_close_lag_7[1:nrow(train_data), "Close"],
                      nd_v_l_32 = nasdaq_vol_lag_32[1:nrow(train_data), "Volume"],
                      sin_45,
                      cos_45, 
                      sin_30,
                      cos_30,
                      sin_22.5,
                      cos_22.5,
                      sin_15,
                      cos_15,
                      sin_11.25,
                      cos_11.25,
                      Close = train_data[, "Close"])
#
  train_data_temp <- train_data
#
# scale data
#
  scale_method <- "0_to_1"
  # scale_method <- "mean_sd"
  if (scale_method == "0_to_1") {
    centers <- apply(train_data_temp, 2, min, na.rm = TRUE)
    scales <- apply(train_data_temp, 2, max, na.rm = TRUE) - centers
  } else if (scale_method == "mean_sd") {
    centers <- apply(train_data_temp, 2, mean, na.rm = TRUE)
    scales <- apply(train_data_temp, 2, sd, na.rm = TRUE)
  }
  train_data_temp <- scale(train_data_temp,
                           center = centers,
                           scale = scales)
  train_data_temp <- train_data_temp[!(is.na(train_data_temp[, "Close"])), ]
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
  activations <- rep(activations_avail[9], 10)
  par_list <- list(activations = activations)
#  
  loss_functions <- init_losses()
  which_losses <- c(10)
  loss_function_used <- loss_functions[which_losses]
  par_list <- c(par_list, loss_function_used = loss_function_used)
#
  unit_structures <- init_units()
  which_units <- c(3)
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
  train_val_splits <- c(0.85)
  par_list <- c(par_list, train_val_splits = train_val_splits)
#
  learning_rates <- c(0.1)
  par_list <- c(par_list, learning_rates = learning_rates)
#  
  decays <- c(0.0)
  par_list <- c(par_list, decays = decays)
#  
  epochs_used <- c(25)
  par_list <- c(par_list, epochs_used = epochs_used)
#
  l1_factors <- c(0.0)
  l2_factors <- c(0.0)
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
  stopping_var <- ""
  change_threshold <- 0
#
# configure ealy stopping
#
  use_early_stopping <- TRUE
  if (use_early_stopping) {
    stopping_var <- "val_loss"
    change_threshold <- 0.001
#
# don't set paitence less than 3 or it can crash peak finding
#
    patience <- 10
    par_list <- c(par_list, stopping_var = stopping_var, 
                  change_threshold = change_threshold, 
                  patience = patience)
    callbacks <- list(callback_early_stopping(monitor = stopping_var,
                                              min_delta = change_threshold,
                                              patience = patience,
                                              mode = "auto"))
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
  pdf_plots <- TRUE
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
  replicates <- 10
  pass <- 0
  prior_pass <- 0
  load_model <- TRUE
  model_file <- "2018-09-07-20-19-1536373150_crypto_fcst_model.hdf5"
  decode_current_model <- TRUE
  boxplots <- TRUE
  save_results_fine <- FALSE
  save_results_summary <- TRUE
#
# only sgd and RMSprop are available at present
#
  optimizer <- "sgd"
  run_date <- as.character(Sys.time(), "%Y-%m-%d")
#
  configuration <- 
    paste0("optimized model using lagged nasdaq and bitcoin with sin-cos periodic features", 
           paste0(unique(activations), " "), "activations")
#
# construct a test set from the last 7 days of data
#
  test_split <- 7 / nrow(train_data_temp)
  test_indices <- seq(floor(nrow(train_data_temp) * (1 - test_split)),
                      nrow(train_data_temp), 1)
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
                      target_col <- which(colnames(train_data_temp) == "Close")
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
                      if (load_model) {
                        model <- load_model_hdf5(model_file, 
                                                 custom_objects = NULL, 
                                                 compile = TRUE)
                      } else {
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
                        if (optimizer == "sgd") {
                          optimizer_used <- 
                            optimizer_sgd(lr = learning_rates[learning_rate_index],
                                          decay = decays[decay_index])
                        } else {
                          optimizer_used <- 
                            optimizer_rmsprop(lr = learning_rates[learning_rate_index])
                        }
#
                        model %>% compile(
                          optimizer = 
                            optimizer_used,
                          loss = loss_function_used,
                          metrics = "mean_absolute_error"
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
                      }
#
                      optimizer_string <- paste0("optimizer = optimizer_",
                                                 optimizer, "(lr = ",
                                                 learning_rates[learning_rate_index], 
                                                 ", decay = ",
                                                 decays[decay_index], ")")
                      train_predictions <-
                        predict(model, list(x_train))
#
                      val_predictions <- 
                        predict(model, list(x_val))
                      val_metrics <- get_metrics(val_predictions, y_val)
                      val_MAE <- val_metrics[[1]]
#
                      test_predictions <- 
                        predict(model, list(x_test))
#
                      x_summary <- 
                        rbind(cbind(x_train, data_src = rep("train", nrow(x_train))),
                              cbind(x_val, data_src = rep("val", nrow(x_val))),
                              cbind(x_test, data_src = rep("test", nrow(x_test))))
                      x_summary <- 
                        as.matrix(x_summary[order(x_summary[, "Date"], 
                                                  decreasing = FALSE), ])
                      summary_predictions <- 
                        data.frame(pred = predict(model, x_summary[, 1:(ncol(x_summary) - 1)]))
                      pred_summary <- as.data.frame(cbind(x_summary, summary_predictions))
                      targets <- rbind(train_data_temp, test_data)[, "Close"]
                      pred_summary <- as.data.frame(cbind(pred_summary, actual = targets))
#                      
# scale the values for plotting
#                      
                      pred_summary[, "Date"] <- 
                        as.numeric(as.character(pred_summary[, "Date"]))
                      pred_summary[, "pred"] <-
                        as.numeric(as.character(pred_summary[, "pred"]))
                      pred_summary[, "actual"] <-
                        as.numeric(as.character(pred_summary[, "actual"]))
                      for (i in 1:nrow(pred_summary)) {
                        pred_summary[i, "Date"] <- 
                          as.numeric(pred_summary[i, "Date"]) *
                          scales["Date"] + centers["Date"]
                        pred_summary[i, "pred"] <- 
                          as.numeric(pred_summary[i, "pred"]) *
                          scales["Close"] + centers["Close"]
                        pred_summary[i, "actual"] <- 
                          as.numeric(pred_summary[i, "actual"]) *
                          scales["Close"] + centers["Close"]
                      }
                      pred_summary[, "Date"] <-
                        as.Date(pred_summary[, "Date"], origin = '1970-01-01')
#
# summary stats to add to the plot
#                      
                      test_MAE <- 
                        get_metrics(pred_summary[pred_summary[, "data_src"] == "test",
                                                 "pred"],
                                    pred_summary[pred_summary[, "data_src"] == "test",
                                                 "actual"])[["MAE"]]
                      test_MAPE <- 100 * test_MAE /
                        mean(pred_summary[pred_summary[, "data_src"] == "test", "actual"])
                      label_x <- 
                        min(as.numeric(pred_summary[, "Date"])) / 2 + 
                        max(as.numeric(pred_summary[, "Date"])) / 2 
                      label_x <- as.Date(label_x, origin = '1970-01-01')
                      label_y <- min(pred_summary[, "actual"]) + 
                        0.75 * (max(pred_summary[, "actual"]) - 
                                  min(pred_summary[, "actual"]))
                      MAPE_label <- round(test_MAPE, 2)
                      MAE_label <- round(test_MAE, 0)
#
# visualize predictions
#                      
                      pred_plot <- pred_summary %>%
                        ggplot(aes(x = Date, y = actual, color = data_src)) +
                        geom_point() +
                        geom_line(aes(x = Date, y = pred, group = 1)) +
                        scale_x_date(date_labels = "%Y-%m-%d") +
                        xlab("") +
                        ylab("Bitcoin closing price") +
                        scale_y_continuous(labels = dollar) +
                        annotate(geom = "text", 
                                 x = label_x, y = label_y,
                                 label = paste0("mean % error (test) = ",
                                                MAPE_label,
                                                " %"),
                                 hjust = 0) +
                        annotate(geom = "text", 
                                 x = label_x, y = label_y - 500,
                                 label = paste0("mean error (test) = ",
                                                MAE_label),
                                 hjust = 0) 
                      print(pred_plot)
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
                      if (!(load_model)) {
                        best_epoch_val <- get_peak(history, 
                                                   "val_mean_absolute_error",
                                                   smoothing = 3,
                                                   direction = "min")
#                      
                        max_epoch <- length(history[["metrics"]][["loss"]])
                      }
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
                      experiment_record[pass - prior_pass, ]$which_units <-
                        which_units[unit_index]
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
                      if (!(load_model)) {
                        experiment_record[pass - prior_pass, ]$train_MAE <-
                          history[["metrics"]][["mean_absolute_error"]][max_epoch]
                        experiment_record[pass - prior_pass, ]$train_MAE_at_best_val <-
                          history[["metrics"]][["mean_absolute_error"]][best_epoch_val]
                        experiment_record[pass - prior_pass, ]$val_MAE <-
                          history[["metrics"]][["val_mean_absolute_error"]][max_epoch]
                        experiment_record[pass - prior_pass, ]$best_epoch_val <-
                          best_epoch_val
                        experiment_record[pass - prior_pass, ]$best_MAE_val <-
                          history[["metrics"]][["val_mean_absolute_error"]][best_epoch_val]
                      } else {
                        experiment_record[pass - prior_pass, ]$val_MAE <-
                          val_MAE
                      }
                      experiment_record[pass - prior_pass, ]$test_MAE <-
                        test_MAE
                      if (!(load_model)) {
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
                        ylim_1 <- 
                          c(max(0, 0.9 * min(min(history[["metrics"]][["loss"]]),
                                             min(history[["metrics"]][["val_loss"]]))),
                            1.1 * max(max(history[["metrics"]][["loss"]]),
                                      max(history[["metrics"]][["val_loss"]])))
                        ylim_1 <- 
                          c(floor(10 * ylim_1[1]) / 10,
                            round(10 * ylim_1[2] / 10, 1))
                        plot(history[["metrics"]][["val_loss"]],
                             type = "l",
                             col = "blue",
                             lwd = 0.5,
                             yaxt = "n",
                             ylab = "",
                             ylim = ylim_1,
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
                        ylim_2 <- 
                          c(max(0, 0.9 * min(min(history[["metrics"]][["mean_absolute_error"]]),
                                             min(history[["metrics"]][["val_mean_absolute_error"]]))),
                            1.1 * max(max(history[["metrics"]][["mean_absolute_error"]]),
                                      max(history[["metrics"]][["val_mean_absolute_error"]])))
                        ylim_2 <- 
                          c(floor(10 * ylim_2[1]) / 10,
                            round(10 * ylim_2[2] / 10, 1))
                        plot(history[["metrics"]][["val_mean_absolute_error"]],
                             type = "l",
                             col = "red",
                             lwd = 0.5,
                             ylim = ylim_2,
                             ylab = "",
                             xaxt = "n",
                             xlab = "")
                        grid(lty = 1, lwd = 0.5)
                        lines(c(history[["metrics"]][["mean_absolute_error"]]),
                              col = "purple",
                              lwd = 0.5)
                        mtext(side = 4, 
                              line = 2.5, 
                              at = (ylim_2[1] + ylim_2[2]) / 2 - 
                                0.09 * (ylim_2[2] - ylim_2[1]), 
                              "val", col = "blue")
                        mtext(side = 4, 
                              line = 2.5, 
                              at = (ylim_2[1] + ylim_2[2]) / 2 + 
                                0.08 * (ylim_2[2] - ylim_2[1]),
                              "     / train loss (mse)", 
                              col = "darkgreen")
                        mtext(side = 2, 
                              line = 2.5, 
                              at = (ylim_2[1] + ylim_2[2]) / 2 - 
                                0.07 * (ylim_2[2] - ylim_2[1]),
                              "train", 
                              col = "purple")
                        mtext(side = 2, 
                              line = 2.5,
                              at = (ylim_2[1] + ylim_2[2]) / 2 + 
                                0.07 * (ylim_2[2] - ylim_2[1]),
                              "     / val MAE", 
                              col = "red")
                      }
                    }
                  }
                }
                if (save_results_fine) {
                  time_stamp_fine <- 
                    as.character(Sys.time(), "%Y-%m-%d-%H-%M-%s")
                  write.csv(experiment_record, paste0(time_stamp_fine, 
                                                      "_nn_experiment_record.csv"))
                }
                if (prior_pass == 0) {
                  experiment_records <- experiment_record
                } else {
                  experiment_records <- rbind(experiment_records,
                                              experiment_record)
                }
                if (!(load_model)) {
                  if (pass == 0) {
                    save_model_hdf5(model, 
                                    paste0(time_stamp, 
                                           "_crypto_fcst_model.hdf5"), 
                                    overwrite = TRUE,
                                    include_optimizer = TRUE)
                  } else {
                    if (experiment_records[, "val_MAE"] <=
                        min(experiment_records[, "val_MAE"])) {
                      save_model_hdf5(model, 
                                      paste0(time_stamp, 
                                             "_crypto_fcst_model.hdf5"), 
                                      overwrite = TRUE,
                                      include_optimizer = TRUE)
                    }
                  }
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
  }
#
# convert layer text to numeric for charting
#
  make_layers_numeric(experiment_records, mod_layers)
#
  train_cutoff <- 0.0
  val_cutoff <- 0.0
  test_cutoff <- 0.0
#
  if (boxplots) {
    boxplot_results(experiment_records, 
                    dep_var = "val_MAE", 
                    exclude_below = val_cutoff, 
                    par_list)
#
    boxplot_results(experiment_records, 
                    dep_var = "test_MAE", 
                    exclude_below = test_cutoff, 
                    par_list)
  }
  if (pdf_plots) {
    while (!is.null(dev.list())) {dev.off()}
  }
#
  show_summary(par_list = par_list, experiment_records = experiment_records)
  