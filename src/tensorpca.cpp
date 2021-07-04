#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

// Generates a rank-1 tensor as lambda * a \otimes b \otimes c.
// [[Rcpp::export]]
arma::cube GetRank1Tensor(double lambda,
                          const arma::vec& a,
                          const arma::vec& b,
                          const arma::vec& c) {
    int d1 = a.size();
    int d2 = b.size();
    int d3 = c.size();
    arma::cube ret(d1, d2, d3);
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                ret(i, j, k) = lambda * a(i) * b(j) * c(k);
            }
        }
    }
    return ret;
}

// Updates a in power iteration.
// input_tensor is input tensor. eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2 store factors of decomposed tensor at each
// iteration step. dims are dimensions of input_tensor. m is the current
// number of iteration. r is the current number of factors.
void PowerIterUpdateA(arma::cube* input_tensor,
                      arma::cube* eigen_vecs_d0,
                      arma::cube* eigen_vecs_d1,
                      arma::cube* eigen_vecs_d2,
                      const Rcpp::IntegerVector& dims,
                      int m, int r) {
    for (int i = 0; i < dims[0]; ++i) {
        (*eigen_vecs_d0)(i, r, m) = 0.0;
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                (*eigen_vecs_d0)(i, r, m) += (*input_tensor)(i, j, k) * (*eigen_vecs_d1)(j, r, m - 1) * (*eigen_vecs_d2)(k, r, m - 1);
            }
        }
    }
    double len = norm(eigen_vecs_d0->slice(m).col(r), 2);
    for (int i = 0; i < dims[0]; ++i) {
        (*eigen_vecs_d0)(i, r, m) /= len;
    }
}

// Updates b in power iteration.
// input_tensor is input tensor. eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2 store factors of decomposed tensor at each
// iteration step. dims are dimensions of input_tensor. m is the current
// number of iteration. r is the current number of factors. sigma is the std
// of idiosyncratic noise.
void PowerIterUpdateB(arma::cube* input_tensor,
                      arma::cube* eigen_vecs_d0,
                      arma::cube* eigen_vecs_d1,
                      arma::cube* eigen_vecs_d2,
                      const Rcpp::IntegerVector& dims,
                      int m, int r, double sigma) {
    for (int j = 0; j < dims[1]; ++j) {
        (*eigen_vecs_d1)(j, r, m) = 0.0;
        for (int i = 0; i < dims[0]; ++i) {
            for (int k = 0; k < dims[2]; ++k) {
                (*eigen_vecs_d1)(j, r, m) += (
                        (*input_tensor)(i, j, k) * (*eigen_vecs_d0)(i, r, m) * (*eigen_vecs_d2)(k, r, m - 1));
            }
        }
        (*eigen_vecs_d1)(j, r, m) -= sigma * sigma * (*eigen_vecs_d1)(j, r, m - 1);
    }
    double len = arma::norm(eigen_vecs_d1->slice(m).col(r), 2);
    for (int j = 0; j < dims[1]; ++j) {
        (*eigen_vecs_d1)(j, r, m) /= len;
    }
}

// Updates c in power iteration.
// input_tensor is input tensor. eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2 store factors of decomposed tensor at each
// iteration step. dims are dimensions of input_tensor. m is the current
// number of iteration. r is the current number of factors. sigma is the std
// of idiosyncratic noise.
void PowerIterUpdateC(arma::cube* input_tensor,
                      arma::cube* eigen_vecs_d0,
                      arma::cube* eigen_vecs_d1,
                      arma::cube* eigen_vecs_d2,
                      const Rcpp::IntegerVector& dims,
                      int m, int r, double sigma) {
    for (int k = 0; k < dims[2]; ++k) {
        (*eigen_vecs_d2)(k, r, m) = 0.0;
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                (*eigen_vecs_d2)(k, r, m) += (
                        (*input_tensor)(i, j, k) * (*eigen_vecs_d0)(i, r, m) * (*eigen_vecs_d1)(j, r, m - 1));
            }
        }
        (*eigen_vecs_d2)(k, r, m) -= sigma * sigma * (*eigen_vecs_d2)(k, r, m - 1);
    }
    double len = arma::norm(eigen_vecs_d2->slice(m).col(r), 2);
    for (int k = 0; k < dims[2]; ++k) {
        (*eigen_vecs_d2)(k, r, m) /= len;
    }
}

// Updates lambda in power iteration.
// mth column of eigen_vals is the eigenvalues of decomposed tensors at iteration m.
// input_tensor is input tensor. eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2 store factors of decomposed tensor at each
// iteration step. m is the current
// number of iteration. r is the current number of factors.
void UpdateLambda(arma::mat* eigen_vals,
                  const arma::cube& input_tensor,
                  const arma::cube& eigen_vecs_d0,
                  const arma::cube& eigen_vecs_d1,
                  const arma::cube& eigen_vecs_d2,
                  int m, int r) {
    (*eigen_vals)(r, m) =
            arma::accu(
                    input_tensor % GetRank1Tensor(1.0,
                                                  eigen_vecs_d0.slice(m).col(r),
                                                  eigen_vecs_d1.slice(m).col(r),
                                                  eigen_vecs_d2.slice(m).col(r))
            );
}

// Tensor power iteration.
void PowerIteration(arma::cube* input_tensor,
                    arma::mat* eigen_vals,
                    arma::cube* eigen_vecs_d0,
                    arma::cube* eigen_vecs_d1,
                    arma::cube* eigen_vecs_d2,
                    const Rcpp::IntegerVector& dims,
                    int m, int r, double sigma) {
    PowerIterUpdateA(
            input_tensor, eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2, dims, m, r);
    PowerIterUpdateB(
            input_tensor, eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2, dims, m, r, sigma);
    PowerIterUpdateC(
            input_tensor, eigen_vecs_d0, eigen_vecs_d1, eigen_vecs_d2, dims, m, r, sigma);
    UpdateLambda(
            eigen_vals, *input_tensor, *eigen_vecs_d0, *eigen_vecs_d1, *eigen_vecs_d2, m, r);
}

// Gets orthogonal complement of tensor input_tensor w.r.t u \otimes v \otimes w.
void GetOrthComp(arma::cube* input_tensor,
                 const Rcpp::IntegerVector& dims,
                 const arma::vec& u,
                 const arma::vec& v,
                 const arma::vec& w) {
    arma::mat I1 = arma::diagmat(arma::ones<arma::vec>(dims[0]));
    arma::mat I2 = arma::diagmat(arma::ones<arma::vec>(dims[1]));
    arma::mat I3 = arma::diagmat(arma::ones<arma::vec>(dims[2]));
    arma::vec normalized_u = u / arma::norm(u, 2);
    arma::vec normalized_v = v / arma::norm(v, 2);
    arma::vec normalized_w = w / arma::norm(w, 2);
    for (int i = 0; i < dims[2]; ++i) {
        input_tensor->slice(i) = (
                (I1 - normalized_u * (normalized_u.t())) * input_tensor->slice(i));
    }
    for (int i = 0; i < dims[2]; ++i) {
        input_tensor->slice(i) = (
                ((I2 - normalized_v * normalized_v.t()) * (input_tensor->slice(i).t())).t());
    }
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            input_tensor->tube(i, j) = (
                    (I3 - normalized_w * normalized_w.t()) * static_cast<arma::vec>(input_tensor->tube(i, j)));
        }
    }
}

// Conducts 3D tensor decomposition according to
// https://arxiv.org/pdf/1702.07449.pdf. X is the input 3D tensor. tensor_rank
// is the rank of decomposed tensor. sigma is the std of idiosyncratic
// noise. num_power_iter is the number of power iterations. 
// use_unfolding_init specifies whether to initialize power iteration by
// SVD on tensor unfolding. If not, the initialization is from standard gaussian
// distribution.
// [[Rcpp::export]]
Rcpp::List TensorPCA(SEXP X,
                     int tensor_rank,
                     double sigma,
                     int num_power_iter,
                     bool use_unfolding_init=true) {
    Rcpp::NumericVector input_vector(X);
    const Rcpp::IntegerVector dims = input_vector.attr("dim");
    arma::cube input_tensor(
            input_vector.begin(), dims[0], dims[1], dims[2], /*copy_aux_mem=*/false);

    // Creates place holders for decomposed tensor eigen values and
    // eigen vectors at each power iteration step.
    // num_power_iter + 1 is because we store initial guess at 0.
    arma::mat eigen_vals(tensor_rank, num_power_iter + 1, arma::fill::zeros);
    arma::cube eigen_vecs_d0(dims[0], tensor_rank, num_power_iter + 1);
    arma::cube eigen_vecs_d1(dims[1], tensor_rank, num_power_iter + 1);
    arma::cube eigen_vecs_d2(dims[2], tensor_rank, num_power_iter + 1);

    input_tensor.reshape(dims[0], dims[1] * dims[2], 1);
    arma::mat unfolded_tensor = input_tensor.slice(0);
    input_tensor.reshape(dims[0], dims[1], dims[2]);
    arma::mat eigen_vecs_d0_init(dims[0], tensor_rank, /*fill_type=*/arma::fill::randn);
    arma::mat eigen_vecs_d1_init(dims[1], tensor_rank, /*fill_type=*/arma::fill::randn);
    arma::mat eigen_vecs_d2_init(dims[2], tensor_rank, /*fill_type=*/arma::fill::randn);
    arma::cube current_tensor(input_tensor);

    for (int r = 0; r < tensor_rank; ++r) {
        current_tensor.reshape(dims[0], dims[1] * dims[2], 1);
        unfolded_tensor = current_tensor.slice(0);
        current_tensor.reshape(dims[0], dims[1], dims[2]);

        // Obtains initial guess for v and w.
        if (use_unfolding_init) {
            arma::mat first_svd_left_singular_vecs;
            arma::vec first_svd_singular_vals;
            arma::mat first_svd_right_singular_vecs;
            arma::svd_econ(
                    first_svd_left_singular_vecs,
                    first_svd_singular_vals,
                    first_svd_right_singular_vecs,
                    unfolded_tensor, /*mode=*/"both", /*method=*/"dc");
            arma::mat second_svd_left_singular_vecs;
            arma::vec second_svd_singular_vals;
            arma::mat second_svd_right_singular_vecs;
            arma::svd(
                    second_svd_left_singular_vecs,
                    second_svd_singular_vals,
                    second_svd_right_singular_vecs,
                    reshape(
                            arma::mat(first_svd_right_singular_vecs.col(0)),
                            dims[1], dims[2]));
            eigen_vals(r, 0) = first_svd_singular_vals(0) * second_svd_singular_vals(0);
            eigen_vecs_d0_init.col(r) = first_svd_left_singular_vecs.col(0);
            eigen_vecs_d1_init.col(r) = second_svd_left_singular_vecs.col(0);
            eigen_vecs_d2_init.col(r) = second_svd_right_singular_vecs.col(0);
        }
        eigen_vecs_d0.slice(0) = eigen_vecs_d0_init;
        eigen_vecs_d1.slice(0) = eigen_vecs_d1_init;
        eigen_vecs_d2.slice(0) = eigen_vecs_d2_init;

        // Conduct power iteration  .
        for (int m = 1; m <= num_power_iter; ++m) {
            PowerIteration(
                    &current_tensor, &eigen_vals,
                    &eigen_vecs_d0, &eigen_vecs_d1, &eigen_vecs_d2,
                    dims, m, r, sigma);
        }
        current_tensor = (
                current_tensor -
                GetRank1Tensor(
                        eigen_vals(r, num_power_iter),
                        eigen_vecs_d0.slice(num_power_iter).col(r),
                        eigen_vecs_d1.slice(num_power_iter).col(r),
                        eigen_vecs_d2.slice(num_power_iter).col(r)));

        GetOrthComp(&current_tensor, dims,
                    eigen_vecs_d0.slice(num_power_iter).col(r),
                    eigen_vecs_d1.slice(num_power_iter).col(r),
                    eigen_vecs_d2.slice(num_power_iter).col(r));
    }

    return Rcpp::List::create(Rcpp::Named("D") = eigen_vals, Rcpp::Named("U") = eigen_vecs_d0,
                              Rcpp::Named("V") = eigen_vecs_d1, Rcpp::Named("W") = eigen_vecs_d2);
}