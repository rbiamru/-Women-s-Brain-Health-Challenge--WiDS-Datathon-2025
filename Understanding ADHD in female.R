
# Full Pipeline: PCA - Merge - Split - Preprocess - Models - Evaluation


library(data.table)
library(irlba)
library(caret)
library(randomForest)
library(pROC)
library(class)
library(ggplot2) 


# 2. PCA on functional data (reduce ~19,900 -> 60)

func <- fread("D:/PA/STAT8031 Multivariate Statistics/Group Project2/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
func_mat <- as.matrix(func[, !"participant_id", with=FALSE])

#Grouping and Un-supervised learning(PCA) to reduce variables
set.seed(42)
pca100 <- irlba(func_mat, nv = 100)

# Scree & cumulative variance data.frame
eigenvalues        <- pca100$d^2
variance_explained <- eigenvalues / sum(eigenvalues)
cum_variance       <- cumsum(variance_explained)
scree_df <- data.frame(
  Component         = 1:100,
  Eigenvalue        = eigenvalues,
  VarianceExplained = variance_explained,
  CumulativeVar     = cum_variance
)

# Scree plot + cumulative variance
ggplot(scree_df, aes(x = Component)) +
  geom_line(aes(y = Eigenvalue)) +
  geom_point(aes(y = Eigenvalue)) +
  geom_vline(xintercept = 100, linetype = "dashed") +
  labs(title = "Scree Plot", x = "PC", y = "Eigenvalue") +
  theme_minimal()

ggplot(scree_df, aes(x = Component)) +
  geom_line(aes(y = CumulativeVar)) +
  geom_point(aes(y = CumulativeVar)) +
  geom_vline(xintercept = 100, linetype = "dashed") +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "blue") +
  labs(title = "Cumulative Variance Explained", x = "PC", y = "Cumulative Proportion") +
  theme_minimal()
# As per variance graph 
# Scree Plot (Eigenvalues)
# The very first component has an enormous eigenvalue (captures the single biggest axis of variance).
# After about PC5–10, the eigenvalues have already dropped into the “flat tail” near zero.
# By PC20, you’re already deep in that flat region where each additional component is adding very little unique variance.
# Cumulative Variance Explained
# PC1 alone captures around 65% of the variance.
# By PC20, about 85–87% of the total variance gets explained.
# By PC50, around 92–93% gets explained.
# By PC100, around ~97% (as indicated by the dashed line) gets explained.
# Build the 60‐PC table
set.seed(42)
pca60 <- irlba(func_mat, nv = 60)
pcs_dt <- data.table(
  participant_id = func$participant_id,
  pca60$u %*% diag(pca60$d)
)
setnames(pcs_dt, old = paste0("V",1:60), new = paste0("PC",1:60))
rm(func, func_mat); gc()


# 3. Read & key the other tables

sol   <- fread("D:/PA/STAT8031 Multivariate Statistics/Group Project2/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.csv")
cat   <- fread("D:/PA/STAT8031 Multivariate Statistics/Group Project2/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.csv")
quant <- fread("D:/PA/STAT8031 Multivariate Statistics/Group Project2/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.csv")

setkey(sol,   participant_id)
setkey(cat,   participant_id)
setkey(quant, participant_id)
setkey(pcs_dt, participant_id)


# 4. Fast chained join

train_dt <- sol[cat][quant][pcs_dt]
rm(sol, cat, quant, pcs_dt); gc()


# 5. Convert to data.frame and select features

train_df <- as.data.frame(train_dt)

# EDA of Distribution of ADHD_Outcome × Sex_F

cat("\nDistribution of ADHD_Outcome × Sex_F:\n")
print(table(train_df$ADHD_Outcome, train_df$Sex_F))
# Inference:
# very imbalanced in both and sex and AHDH
# males are more prone to ADH
# a statistically significant association between Sex (Male/Female) and ADHD diagnosis.
# Drop ID & the other target (Sex_F), keep only ADHD_Outcome
train_df$participant_id <- NULL
train_df$Sex_F         <- NULL
# Ensure target is a factor
train_df$ADHD_Outcome <- factor(train_df$ADHD_Outcome, levels = c(0,1))


# 6. Train/Test split (70/30 stratified)

set.seed(123)
train_idx <- createDataPartition(train_df$ADHD_Outcome, p = 0.7, list = FALSE)
train <- train_df[train_idx, ]
test  <- train_df[-train_idx, ]

# Separate features and target
y_train <- train$ADHD_Outcome
y_test  <- test$ADHD_Outcome
train_features <- train[,-which(names(train)=="ADHD_Outcome")]
test_features <- test[,-which(names(test)=="ADHD_Outcome")]


# 7. Preprocess: Median Impute, dummy-encode factors & scale numerics

# a) Impute missing values using median
preproc_impute <- preProcess(train_features, method = "medianImpute")
X_train_imputed <- predict(preproc_impute, train_features)
X_test_imputed  <- predict(preproc_impute, test_features)

# b) Dummy-encode all predictors
dummies <- dummyVars(~ ., data = as.data.frame(X_train_imputed))
X_train <- predict(dummies, newdata = as.data.frame(X_train_imputed))
X_test  <- predict(dummies, newdata = as.data.frame(X_test_imputed))

# c) Standardize (center & scale) using train parameters
preproc_scale <- preProcess(X_train, method = c("center","scale"))
X_train_scaled <- predict(preproc_scale, X_train)
X_test_scaled  <- predict(preproc_scale, X_test)


# 8. Model 1: K-Nearest Neighbors (KNN)

# Find the optimal k using cross-validation on the training data
ctrl <- trainControl(method = "cv", number = 5) # 5-fold cross-validation
knn_fit <- train(
  x = X_train_scaled,
  y = y_train,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10 # Trying 10 different values of k
)

# Best k value
best_k <- knn_fit$bestTune$k
print(paste("Optimal k for KNN:", best_k))

# Predict on the test set using the best k
p_knn_cls <- factor(knn(train = X_train_scaled, 
                        test = X_test_scaled,
                        cl = y_train,
                        k = best_k),
                    levels = c(0,1))
p_knn_prob <- predict(knn_fit, newdata = X_test_scaled, type = "prob")[, "1"]

# 9. Model 2: Logistic Regression

glm_df    <- data.frame(X_train_scaled, ADHD_Outcome = y_train)
model_glm <- glm(ADHD_Outcome ~ ., 
                 data = glm_df, 
                 family = binomial)
p_glm_prob <- predict(model_glm, newdata = as.data.frame(X_test_scaled), type = "response")
p_glm_cls   <- factor(ifelse(p_glm_prob > 0.5, 1, 0), levels = c(0,1))


# 10. Model 3: Random Forest

set.seed(123)
model_rf <- randomForest(x = X_train_scaled, 
                         y = y_train, ntree = 500, 
                         importance = TRUE)
p_rf_prob <- predict(model_rf, X_test_scaled, type = "prob")[, "1"]
p_rf_cls   <- predict(model_rf, X_test_scaled)


# 11. Evaluate all three models

eval_model <- function(true, pred_cls, pred_prob) {
  cm  <- confusionMatrix(pred_cls, true, positive = "1")
  roc <- roc(as.numeric(as.character(true)), pred_prob)
  data.frame(
    Accuracy = cm$overall["Accuracy"],
    F1      = cm$byClass["F1"],
    AUC     = as.numeric(auc(roc))
  )
}

res_glm <- eval_model(y_test, p_glm_cls, p_glm_prob)
res_rf  <- eval_model(y_test, p_rf_cls,  p_rf_prob)
res_knn <- eval_model(y_test, p_knn_cls, p_knn_prob)

results <- rbind(
  cbind(Model = "Logistic Regression", res_glm),
  cbind(Model = "Random Forest",       res_rf),
  cbind(Model = "K-Nearest Neighbors", res_knn)
)
print(results)


# 12. Confusion Matrices for Inference

cat("\nConfusion Matrix - Logistic Regression:\n")
cm_glm <- confusionMatrix(p_glm_cls, y_test, positive = "1")
print(cm_glm)

cat("\nConfusion Matrix - Random Forest:\n")
cm_rf <- confusionMatrix(p_rf_cls, y_test, positive = "1")
print(cm_rf)

cat("\nConfusion Matrix - K-Nearest Neighbors:\n")
cm_knn <- confusionMatrix(p_knn_cls, y_test, positive = "1")
print(cm_knn)

# Inference based on Confusion Matrices
cat("\nInference based on Confusion Matrices:\n")
cat("Analyzing the confusion matrices helps understand the types of errors each model makes.\n")
cat("Your inference is that False Negatives (FN) are a bigger problem than False Positives (FP).\n")
cat("Let's look at the counts in the confusion matrices:\n\n")

cat("Logistic Regression: FN =", cm_glm$table["0", "1"], ", FP =", cm_glm$table["1", "0"], "\n")
cat("Random Forest:       FN =", cm_rf$table["0", "1"], ", FP =", cm_rf$table["1", "0"], "\n")
cat("K-Nearest Neighbors: FN =", cm_knn$table["0", "1"], ", FP =", cm_knn$table["1", "0"], "\n\n")

cat("By comparing the FN and FP counts for each model, you can see which models align with your preference of minimizing False Negatives over False Positives for this specific problem.\n")
cat("A higher FN count means the model is more likely to miss individuals who actually have ADHD (label 1).\n")
cat("A higher FP count means the model is more likely to incorrectly classify individuals without ADHD (label 0) as having it.\n")

# Function to create a heatmap from a confusion matrix
plot_confusion_matrix <- function(cm, title = "Confusion Matrix") {
  # Convert confusion matrix to a data frame for ggplot
  cm_df <- as.data.frame(cm$table)
  names(cm_df) <- c("Predicted", "Actual", "Freq")
  
  ggplot(data = cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(height = 0.9, width = 0.9) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5, color = "black", size = 5) +
    theme_minimal() +
    labs(title = title, fill = "Frequency") +
    theme(plot.title = element_text(hjust = 0.5))
}

# Get the confusion matrix objects we already created
cm_glm <- confusionMatrix(p_glm_cls, y_test, positive = "1")
cm_rf <- confusionMatrix(p_rf_cls, y_test, positive = "1")
cm_knn <- confusionMatrix(p_knn_cls, y_test, positive = "1")

# Create and display the heatmaps
plot_confusion_matrix(cm_glm, title = "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(cm_rf, title = "Confusion Matrix - Random Forest")
plot_confusion_matrix(cm_knn, title = "Confusion Matrix - K-Nearest Neighbors")

# 13. ROC Curves

roc_glm <- roc(as.numeric(as.character(y_test)), p_glm_prob)
roc_rf <- roc(as.numeric(as.character(y_test)), p_rf_prob)
roc_knn <- roc(as.numeric(as.character(y_test)), p_knn_prob)

plot(roc_glm, col = "blue", main = "ROC Curves Comparison", xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(roc_rf, add = TRUE, col = "red")
plot(roc_knn, add = TRUE, col = "green")
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "K-Nearest Neighbors"), col = c("blue", "red", "green"), lty = 1)
