// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// GetRank1Tensor
arma::cube GetRank1Tensor(double lambda, const arma::vec& a, const arma::vec& b, const arma::vec& c);
RcppExport SEXP _tensorpca_GetRank1Tensor(SEXP lambdaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP cSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type c(cSEXP);
    rcpp_result_gen = Rcpp::wrap(GetRank1Tensor(lambda, a, b, c));
    return rcpp_result_gen;
END_RCPP
}
// TensorPCA
Rcpp::List TensorPCA(SEXP X, int tensor_rank, double sigma, int num_power_iter, bool use_unfolding_init);
RcppExport SEXP _tensorpca_TensorPCA(SEXP XSEXP, SEXP tensor_rankSEXP, SEXP sigmaSEXP, SEXP num_power_iterSEXP, SEXP use_unfolding_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type tensor_rank(tensor_rankSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< int >::type num_power_iter(num_power_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_unfolding_init(use_unfolding_initSEXP);
    rcpp_result_gen = Rcpp::wrap(TensorPCA(X, tensor_rank, sigma, num_power_iter, use_unfolding_init));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tensorpca_GetRank1Tensor", (DL_FUNC) &_tensorpca_GetRank1Tensor, 4},
    {"_tensorpca_TensorPCA", (DL_FUNC) &_tensorpca_TensorPCA, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_tensorpca(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
