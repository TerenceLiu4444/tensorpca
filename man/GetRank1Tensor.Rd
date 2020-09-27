\name{GetRank1Tensor}
\alias{GetRank1Tensor}
\docType{package}
\title{
Get Rank1 Tensor.
}
\description{
Get Rank1 Tensor from a scalar lambda and three vectors.
}
\usage{
     GetRank1Tensor(lambda, a, b, c)
}
\arguments{
\item{lambda}{Scaling factor of the rank-1 tensor.}
\item{a}{First dimension vector.}
\item{b}{Second dimension vector.}
\item{c}{Third dimension vector.}
}
\value{
Return a rank1 tensor with (i,j,k) element as lambda*a[i]*b[j]*c[k].
}
\examples{
library(tensorpca)
lambda = 1
a = c(1,0,0)
b = c(1,0,0)
c = c(1,0,0)
signal_tensor = GetRank1Tensor(lambda, a, b, c)
}
