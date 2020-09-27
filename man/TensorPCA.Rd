\name{TensorPCA}
\alias{TensorPCA}
\docType{package}
\title{
3D Tensor PCA
}
\description{
3D Tensor PCA according to https://arxiv.org/pdf/1702.07449.pdf.
}
\usage{
TensorPCA(data_tensor, decomposed_tensor_rank, sigma, power_iteration)
}
\arguments{
\item{data_tensor}{3D input data tensor with shape (d1, d2, d3).}
\item{decomposed_tensor_rank}{Number of factors to get from the decomposition.}
\item{sigma}{Standard deviation of idiosyncratic noise.}
\item{power_iteration}{Number of power iteration.}
}
\value{
Returns a list with the following attributes:
\item{D}{D has shape (decomposed_tensor_rank, power_iteration+1). The mth column (D[,m]) contains the eigen values obtained at (m-1) iteration.}
\item{U}{U has shape (d1, decomposed_tensor_rank, power_iteration+1). The mth slice (U[,,m]) contains the eigen vectors across 1st dimension obtaind at (m-1) iteration.}
\item{V}{V has shape (d2, decomposed_tensor_rank, power_iteration+1). The mth slice (V[,,m]) contains the eigen vectors across 2nd dimension obtaind at (m-1) iteration.}
\item{W}{W has shape (d3, decomposed_tensor_rank, power_iteration+1). The mth slice (W[,,m]) contains the eigen vectors across 3rd dimension obtaind at (m-1) iteration.}
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
res = TensorPCA(data_tensor, decomposed_tensor_rank, sigma, power_iteration)
}
