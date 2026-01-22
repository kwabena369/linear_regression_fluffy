# Fluffy Linear Regression
linear regression using only NumPy + gradient descent.  

![Loss History](plots/msl.png)

## What this does

Linear regression tries to find the straight line that best fits your data points.

The equation is simple:  
**ŷ = w × x + b**  
- **w** = weight (slope of the line)  
- **b** = bias (y-intercept)

Goal: find the **best w** and **b** so predictions are as close as possible to real y values.

## How I do it

1. Start with **w = 0** and **b = 0** (random start is also fine)
2. For many iterations (here 1000):
   - Make predictions: `ŷ = w * x + b`
   - Calculate error with **Mean Squared Error** (MSE):  
     `loss = mean( (y_true - ŷ)^2 )`
   - Compute gradients (how much to change w and b):
     ```
     dw = (2 / n) * sum( x * (ŷ - y_true) )
     db = (2 / n) * sum( ŷ - y_true )
     ```
   - Update using gradient descent:
     ```
     w ← w - learning_rate * dw
     b ← b - learning_rate * db
     ```
   - When **dw > 0** → current w is too big → we decrease it (and vice versa)  
     Same logic applies to **db**

3. After training → loss should keep dropping (see plot above)

## Example result

Data:  
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
y = [3, 4, 2, 5, 6, 7, 8, 9,10, 12]

After training → prediction for x = 11 is around **12.2** (makes sense looking at the pattern)

## Files

- `fluffy_linear_regression.py` → the class + training loop  
- `plots/msl.png` → loss curve (MSE dropping over iterations)
