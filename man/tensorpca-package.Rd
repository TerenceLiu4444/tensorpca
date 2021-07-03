\name{tensorpca-package}
\alias{tensorpca-package}
\alias{tensorpca}
\docType{package}
\title{
  Conduct 3D Tensor PCA.
}
\description{
  Conduct orthogonal 3D tensor decomposition according to https://arxiv.org/pdf/1702.07449.pdf.
}
\author{
Tianqi Liu

Maintainer: Tianqi Liu <tianqi.terence.liu@gmail.com>
}
\references{
  Liu, Tianqi, Ming Yuan, and Hongyu Zhao. "Characterizing spatiotemporal transcriptome of human brain via low rank tensor decomposition." arXiv preprint arXiv:1702.07449 (2017).
}
\keyword{ package }
\seealso{
  Optional links to other man pages
}
\examples{
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
res = TensorPCA(data_tensor, decomposed_tensor_rank, sigma, power_iteration, use_unfolding_init=TRUE)
print("Eigen values of the decomposed tensor:")
print(res$D[, power_iteration])

print("Eigen values of the signal tensor:")
print(c(1,0,0))

print("Eigen vectors of the decomposed tensor:")
print(res$U[, 1, power_iteration])
    
print("Eigen vectors of the signal tensor:")
print(c(1,0,0))
}
