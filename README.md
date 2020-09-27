# tensorpca
3d tensor principle component analysis as described in https://arxiv.org/abs/1702.07449.

# Example usage

```
install.packages("devtools")
library(devtools)
library(tensorpca)
d = 3
sigma = 0.01
a = c(1,0,0)
b = c(1,0,0)
c = c(1,0,0)

signal_tensor = GetRank1Tensor(1,a,b,c)
data_tensor = signal_tensor + sigma * array(rnorm(d * d * d), c(d, d, d))
power_iteration = 10
decomposed_tensor_rank = 3
res = TensorPCA(data_tensor, decomposed_tensor_rank, sigma, power_iteration)
print("Eigen values of the decomposed tensor:")
print(res$D[, power_iteration])

print("Eigen values of the signal tensor:")
print(c(1,0,0))

print("Eigen vectors of the decomposed tensor:")
print(res$U[, 1, power_iteration])
    
print("Eigen vectors of the signal tensor:")
print(c(1,0,0))
```
