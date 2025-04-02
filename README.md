# Build Your Own Machine Learning Model

## 🤖 Machine Learning Overview

### 🔹 What is Machine Learning?

Machine Learning (ML) is a branch of artificial intelligence (AI) that enables computers to learn patterns from data without being explicitly programmed. It can be broadly classified into three main categories:

1. **Supervised Learning**  
   - Uses labeled data (input-output pairs) to train models.  
   - Examples:  
     - *Linear Regression* (predicting continuous values)  
     - *Classification* (e.g., spam email detection)  

2. **Unsupervised Learning**  
   - Works with unlabeled data (only inputs) to find hidden structures.  
   - Example:  
     - *Clustering* (e.g., customer segmentation)  

3. **Reinforcement Learning**  
   - Involves learning through trial and error by interacting with an environment.  
   - Examples:  
     - Self-driving cars  
     - Game-playing AI  

### 🔹 Prerequisites for Machine Learning

To dive into ML, you’ll need a solid foundation in the following areas:  
- **Mathematics**: Linear Algebra, Probability, and Calculus  
- **Programming**: Python (with libraries like NumPy, Pandas, and Scikit-Learn)  
- **Data Preprocessing**: Handling missing values, feature scaling, and data normalization  
- **Model Evaluation**: Metrics like RMSE (Root Mean Squared Error), Accuracy, and Precision-Recall  

---

# 📌 Linear Regression: A Complete Guide

## 🔹 What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used for predicting a continuous target variable **y** based on an input feature **X**. The relationship between **X** and **y** is represented as:

```
y = mX + b
```

Where:

- **m** → Slope of the line (weight/parameter)
- **b** → Intercept (bias/constant)
- **X** → Input feature (independent variable)
- **y** → Output (dependent variable)

---

## 🔹 Understanding the Cost Function

The **Cost Function** measures the error between actual values (**y**) and predicted values (**ŷ**). We use **Mean Squared Error (MSE):**

```
J(m, b) = (1/N) * Σ (ŷ_i - y_i)^2
```

Where:

- **N** → Number of data points
- **ŷ = mX + b** → Predicted values
- **y** → Actual values

The goal of Linear Regression is to **find the best values of m and b that minimize the cost function**.

---

## 🔹 Gradient Descent: Optimizing m and b

Gradient Descent is an optimization algorithm used to update **m** and **b** iteratively to minimize the cost function.

The **gradients** (derivatives) of the cost function with respect to **m** and **b** are:

```
∂J/∂m = (2/N) * Σ (mX_i + b - y_i) X_i
```

```
∂J/∂b = (2/N) * Σ (mX_i + b - y_i)
```

### 🔹 Parameter Update Equations

We update **m** and **b** using the learning rate (α):

```
m = m - α * (∂J/∂m)
```

```
b = b - α * (∂J/∂b)
```

Where **α (learning rate)** determines how big the update step should be.

---

# 🔹 Example: Learning y = 3X + 5

Let's take **5 data points**:

| X | y (Actual) |
| - | ---------- |
| 1 | 8          |
| 2 | 11         |
| 3 | 14         |
| 4 | 17         |
| 5 | 20         |

### **Step 1: Initialize Parameters**

Let's start with random values:

```
m = 0.5, b = 0.1
α = 0.01 (Learning rate)
```

### **Step 2: Compute Predictions**

Using **ŷ = mX + b**:

| X | y (Actual) | ŷ (Predicted) |
| - | ---------- | ------------- |
| 1 | 8          | 0.6           |
| 2 | 11         | 1.1           |
| 3 | 14         | 1.6           |
| 4 | 17         | 2.1           |
| 5 | 20         | 2.6           |

### **Step 3: Compute Error**

Error = **ŷ - y**:

| X | y (Actual) | ŷ (Predicted) | Error (ŷ - y) |
| - | ---------- | ------------- | ------------- |
| 1 | 8          | 0.6           | -7.4          |
| 2 | 11         | 1.1           | -9.9          |
| 3 | 14         | 1.6           | -12.4         |
| 4 | 17         | 2.1           | -14.9         |
| 5 | 20         | 2.6           | -17.4         |

### **Step 4: Compute Gradients**

```
∂J/∂m = (2/5) * Σ (ŷ_i - y_i) X_i
      = (2/5) * (-211)
      = -84.4
```

```
∂J/∂b = (2/5) * Σ (ŷ_i - y_i)
      = (2/5) * (-62)
      = -24.8
```

### **Step 5: Update Parameters**

```
m = 0.5 - (0.01 * -84.4)
  = 0.5 + 0.844
  = 1.344
```

```
b = 0.1 - (0.01 * -24.8)
  = 0.1 + 0.248
  = 0.348
```

### **Updated Values**

```
m = 1.344, b = 0.348
```

---

## 🔄 Repeat Until Convergence

Repeating this process **1000 times**, the parameters gradually converge to:

```
m ≈ 3, b ≈ 5
```

✅ **Final Model:**

```
y = 3X + 5
```

---

# 🎯 Key Takeaways
