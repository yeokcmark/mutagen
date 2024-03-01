rm(list=ls())
pacman::p_load("tidyverse", #for tidy data science practice
               "tidymodels", "workflows", "finetune", "themis", "embed",# for tidy machine learning
               "pacman", #package manager
               "devtools", #developer tools
               "Hmisc", "skimr", "broom", "modelr",#for EDA
               "jtools", "huxtable", "interactions", # for EDA
               "ggthemes", "ggstatsplot", "GGally",
               "scales", "gridExtra", "patchwork", "ggalt", "vip",
               "ggstance", "ggfortify", # for ggplot
               "DT", "plotly", #interactive Data Viz
               # Lets install some ML related packages that will help tidymodels::
               "usemodels", "poissonreg", "agua", "sparklyr", "dials",#load computational engines
               "doParallel", # for parallel processing (speedy computation)
               "ranger", "xgboost", "glmnet", "kknn", "earth", "klaR", "discrim", "naivebayes", "baguette", "kernlab",#random forest
               "janitor", "lubridate")

df <-read_csv("mutagen.csv")
df %>% count(outcome)
data <-
  df %>% 
  dplyr::select(-1)
# confirm no NA data
table(is.na(data))

## split data

set.seed(2024030101)
data_split <-
  data %>% 
  initial_split(strata = outcome)

data_train <-
  data_split %>% 
  training()
data_test <-
  data_split %>% 
  testing()

data_fold <-
  data_train %>% 
  vfold_cv(v = 10, strata = outcome)

## Recipe

base_rec <-
  recipes::recipe(formula = outcome ~.,
                  data = data_train) %>% 
  step_zv(all_predictors()) %>% # remove zero variance
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())

pca_rec <-
  base_rec %>% 
  step_pca(all_predictors())

## models


# random forest
rf_spec <-
  rand_forest() %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_mode("classification") %>% 
  set_args(trees = 1000L,
           mtry = tune(),
           min_n = tune())

# xgboost
xgb_spec <-
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  set_args(trees = 1000L,
           tree_depth = tune(),
           min_n = tune(),
           loss_reduction = tune(),
           sample_size = tune(),
           mtry = tune(),
           learn_rate = tune(),
           stop_iter = 10)

# Logistic Regression Model
logistic_spec <- 
  logistic_reg() %>%
  set_engine(engine = 'glm') %>%
  set_mode('classification') 

null_spec <-
  null_model() %>% 
  set_mode("classification") %>% 
  set_engine("parsnip")

# workflow set

base_set <- 
  workflow_set (
    list(basic = base_rec,
         pca = pca_rec), #preprocessor
    list(rand_forest = rf_spec,
         xgboost = xgb_spec,
         logistic = logistic_spec,
         null = null_spec), #model
    cross = TRUE) #default is cross = TRUE

# tune hyper parameters
set.seed(2024030302)
ncores <- detectCores() - 1
cl <- makePSOCKcluster(ncores)
doParallel::registerDoParallel(cl)

racing_results <-
  workflow_map(base_set,
               fn = "tune_race_anova",
               resamples = data_fold,
               metrics = metric_set(roc_auc, f_meas, accuracy, mn_log_loss),
               control = control_race(verbose = TRUE,
                                      verbose_elim = TRUE,
                                      allow_par = TRUE,
                                      #save_workflow = TRUE,
                                      parallel_over = "everything"))
racing_results %>% autoplot()
save.image("mutagen_results.RData")
