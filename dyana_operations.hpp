//
// Created by YAN Yuchen on 11/13/2018.
//

#ifndef DYANA_OPERATIONS_HPP
#define DYANA_OPERATIONS_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dyana_common.hpp"


namespace dyana {

/**
 * \ingroup normoperations
 * \brief Weight normalization
 * \details Performs weight normalization :
 *
 * \f$
 * \begin{split}
 *    \hat{w} &= g\frac{w}{\Vert w\Vert}\\
 * \end{split}
 * \f$
 *
 * Reference : [Salimans, Kingma 2016](https://arxiv.org/abs/1602.07868)
 *
 * \param w Input expression (weight parameter)
 * \param g Gain (scalar expression, usually also a parameter)
 * \return An expression of the same dimension as `w`
 */
  inline tensor weight_norm(const tensor &w, const tensor &g) { return dynet::weight_norm(w, g); }

  inline tensor operator+(const tensor &x, const tensor &y) { return dynet::operator+(x, y); }

  inline tensor operator+(const tensor &x, float y) { return dynet::operator+(x, y); }

  inline tensor operator+(const tensor &x, double y) { return dynet::operator+(x, (float)y); }

  inline tensor operator+(const float x, const tensor &y) { return dynet::operator+(x, y); }

  inline tensor operator+(const double x, const tensor &y) { return dynet::operator+((double)x, y); }

  inline tensor operator+(const tensor &x, const parameter &y) { return x + dyana::tensor(y); }

  inline tensor operator+(const parameter &x, const tensor &y) { return dyana::tensor(x) + y; }

  inline tensor operator-(const tensor &x, const tensor &y) { return dynet::operator-(x, y); }

  inline tensor operator-(const tensor &x, float y) { return dynet::operator-(x, y); }

  inline tensor operator-(const tensor &x, double y) { return dynet::operator-(x, (float)y); }

  inline tensor operator-(float x, const tensor &y) { return dynet::operator-(x, y); }

  inline tensor operator-(double x, const tensor &y) { return dynet::operator-((float)x, y); }

  inline tensor operator-(const tensor &x) { return dynet::operator-(x); }

  inline tensor operator-(const tensor &x, const parameter &y) { return x - dyana::tensor(y); }

  inline tensor operator-(const parameter &x, const tensor &y) { return dyana::tensor(x) - y; }

  inline tensor operator*(const tensor &x, const tensor &y) { return dynet::operator*(x, y); }

  inline tensor operator*(const tensor &x, float y) { return dynet::operator*(x, y); }

  inline tensor operator*(const tensor &x, double y) { return dynet::operator*(x, (float)y); }

  inline tensor operator*(float x, const tensor &y) { return dynet::operator*(x, y); }

  inline tensor operator*(double x, const tensor &y) { return dynet::operator*((float)x, y); }

  inline tensor operator*(const tensor &x, const parameter &y) { return x * dyana::tensor(y); }

  inline tensor operator*(const parameter &x, const tensor &y) { return dyana::tensor(x) * y; }

  inline tensor operator/(const tensor &x, const tensor& y) { return dynet::operator/(x, y); }

  inline tensor operator/(const tensor &x, float y) { return dynet::operator/(x, y); }

  inline tensor operator/(const tensor &x, double y) { return dynet::operator/(x, (float)y); }

  inline tensor operator/(const parameter &x, float y) { return dyana::tensor(x) / y; }

  inline tensor operator/(const parameter &x, double y) { return dyana::tensor(x) / (float)y; }

  inline tensor operator/(const tensor &x, const parameter &y) { return x / dyana::tensor(y); }

  inline tensor operator/(const parameter &x, const tensor &y) { return dyana::tensor(x) / y; }

  inline tensor zeros(const Dim &d) { return dynet::zeros(_cg(), d); }

  inline tensor ones(const Dim &d) { return dynet::ones(_cg(), d); }

  /**
   * \ingroup inputoperations
   * \brief Create a random normal vector
   * \details Create a vector distributed according to normal distribution with specified mean and standard deviation.
   *
   * \param d The dimensions of the input
   * \param mean The mean of the distribution (default: 0.0)
   * \param stddev The standard deviation of the distribution (default: 1.0)
   *
   * \return A "d" dimensioned normally distributed vector
   */
  inline tensor random_normal(const Dim& d, float mean = 0, float stddev = 1) { return dynet::random_normal(_cg(), d, mean, stddev);}

  /**
   * \ingroup inputoperations
   * \brief Create a random uniform vector
   * \details Create a vector distributed according to uniform distribution with boundaries left and right.
   *
   * \param d The dimensions of the input
   * \param left The left boundary
   * \param right The right boundary
   *
   * \return A "d" dimensioned uniform distributed vector
   */
  inline tensor random_uniform(const Dim& d, float left, float right) { return dynet::random_uniform(_cg(), d, left, right); }

/**
 * \ingroup arithmeticoperations
 * \brief Affine transform
 * \details This performs an affine transform over an arbitrary (odd) number of expressions
 *          held in the input initializer list xs.
 *          The first expression is the "bias," which is added to the expression as-is.
 *          The remaining expressions are multiplied together in pairs, then added.
 *          A very common usage case is the calculation of the score for a neural network
 *          layer (e.g. b + Wz) where b is the bias, W is the weight matrix, and z is the
 *          input. In this case xs[0] = b, xs[1] = W, and xs[2] = z.
 *
 * \param xs An initializer list containing an odd number of expressions
 *
 * \return An expression equal to: xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...
 */
  inline tensor affine_transform(const std::initializer_list<tensor> &xs) {
    return dynet::affine_transform(tensor::vector_cast_to_base(xs));
  }

  inline tensor affine_transform(const std::vector<tensor> &xs) {
    return dynet::affine_transform(tensor::vector_cast_to_base(xs));
  }

/**
 * \ingroup arithmeticoperations
 * \brief Sum
 * \details This performs an elementwise sum over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to xs[0][i] + xs[1][i] + ...
 */
  inline tensor sum(const std::vector<tensor> &xs) { return dynet::sum(tensor::vector_cast_to_base(xs)); }


/**
 * \ingroup arithmeticoperations
 * \brief Sum all elements
 * \details Sum all the elements in an expression.
 *
 * \param x The input expression
 *
 * \return The sum of all of its elements
 */
  inline tensor sum_elems(const tensor &x) { return dynet::sum_elems(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute moment over all elements
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 * \param r Order of the moment
 *
 * \return A scalar expression (with a potential batch dimension)
 */
  inline tensor moment_elems(const tensor &x, unsigned r) { return dynet::moment_elems(x, r); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
  inline tensor mean_elems(const tensor &x) { return dynet::mean_elems(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute Standard deviation over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
  inline tensor std_elems(const tensor &x) { return dynet::std_elems(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Sum over minibatches
 * \details Sum an expression that consists of multiple minibatches into one of
 *          equal dimension but with only a single minibatch. This is useful
 *          for summing loss functions at the end of minibatch training.
 *
 * \param x The input mini-batched expression
 *
 * \return An expression with a single batch
 */
  inline tensor sum_batches(const tensor &x) { return dynet::sum_batches(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute moment over minibatches
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ along the batch dimension
 *
 * \param x The input mini-batched expression
 * \param r Order of the moment
 *
 * \return An expression with a single batch
 */
  inline tensor moment_batches(const tensor &x, unsigned r) { return dynet::moment_batches(x, r); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension
 *
 * \param x The input mini-batched expression
 *
 * \return An expression with a single batch
 */
  inline tensor mean_batches(const tensor &x) { return dynet::mean_batches(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute standard deviation over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
  inline tensor std_batches(const tensor &x) { return dynet::std_batches(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute sum along a specific dimension or dimensions
 * \details Compute the sum along a specific dimension or dimensions
 *
 * \param x The input mini-batched expression
 * \param d Dimensions along which to reduce
 * \param b Whether to include batch dimension (default: false)
 *
 * \return An expression with |d| less dimensions and possibly dropped batch dimension
 */
  inline tensor sum_dim(const tensor &x, const std::vector<unsigned> &dims, bool b = false) {
    return dynet::sum_dim(x, dims, b);
  }

// These are deprecated but kept for backward compatibility
  inline tensor sum_rows(const tensor &x) { return dynet::sum_rows(x); }

  inline tensor sum_cols(const tensor &x) { return dynet::sum_cols(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute cumulative sum along a specific dimension
 * \details Compute the cumulative sum along a specific dimension: \f$y_i=\sum_{j\leq i}x_j\f$
 *
 * \param x The input mini-batched expression
 * \param d Dimensions along which to compute the cumulative sum
 *
 * \return An expression of the same shape as the input
 */
  inline tensor cumsum(const tensor &x, unsigned d) { return dynet::cumsum(x, d); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute moment along a specific dimension
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ along a specific dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimensions along which to reduce
 * \param r Order of the moment
 * \param b Whether to include batch dimension (default: false)
 * \param n If > 0, overwrite the n in the equation by this value, useful for masking (default: 0)
 *
 * \return An expression with |d| less dimensions and possibly dropped batch dimension
 */
  inline tensor moment_dim(const tensor &x, const std::vector<unsigned> &dims, unsigned r, bool b = false,
                           unsigned n = 0) { return dynet::moment_dim(x, dims, r, b, n); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean along  a specific dimension
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ along a specific dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimensions along which to reduce
 * \param b Whether to include batch dimension (default: false)
 * \param n If > 0, overwrite the n in the equation by this value, useful for masking (default: 0)
 *
 * \return An expression with |d| less dimensions and possibly dropped batch dimension
 */
  inline tensor mean_dim(const tensor &x, const std::vector<unsigned> &dims, bool b = false,
                         unsigned n = 0) { return dynet::mean_dim(x, dims, b, n); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute standard deviation along an arbitrary dimension
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ along an arbitrary dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimensions along which to reduce
 * \param b Whether to include batch dimension (default: false)
 * \param n If > 0, overwrite the n in the equation by this value, useful for masking (default: 0)
 *
 * \return An expression with |d| less dimensions and possibly dropped batch dimension
 */
  inline tensor std_dim(const tensor &x, const std::vector<unsigned> &dims, bool b = false,
                        unsigned n = 0) { return dynet::std_dim(x, dims, b, n); }

/**
 * \ingroup arithmeticoperations
 * \brief Average
 * \details This performs an elementwise average over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to (xs[0][i] + xs[1][i] + ...)/|xs|
 */
  inline tensor average(const std::vector<tensor> &xs) { return dynet::average(tensor::vector_cast_to_base(xs)); }

/**
 * \ingroup arithmeticoperations
 * \brief Square root
 * \details Elementwise square root.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\sqrt(x_i)\f$
 */
  inline tensor sqrt(const tensor &x) { return dynet::sqrt(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Absolute value
 * \details Elementwise absolute value.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\vert x_i\vert\f$
 */
  inline tensor abs(const tensor &x) { return dynet::abs(x); }


/**
 * \ingroup arithmeticoperations
 * \brief Gaussian error function
 * \details Elementwise calculation of the Gaussian error function
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to erf(x_i)
 */
  inline tensor erf(const tensor &x) { return dynet::erf(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse sine
 * \details Elementwise calculation of the inverse sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to asin(x_i)
 */
  inline tensor asin(const tensor &x) { return dynet::asin(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse cosine
 * \details Elementwise calculation of the inverse cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to acos(x_i)
 */
  inline tensor acos(const tensor &x) { return dynet::acos(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse tangent
 * \details Elementwise calculation of the inverse tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to atan(x_i)
 */
  inline tensor atan(const tensor &x) { return dynet::atan(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Sine
 * \details Elementwise calculation of the sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to sin(x_i)
 */
  inline tensor sin(const tensor &x) { return dynet::sin(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Cosine
 * \details Elementwise calculation of the cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to cos(x_i)
 */
  inline tensor cos(const tensor &x) { return dynet::cos(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Tangent
 * \details Elementwise calculation of the tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to tan(x_i)
 */
  inline tensor tan(const tensor &x) { return dynet::tan(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic sine
 * \details Elementwise calculation of the hyperbolic sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to sinh(x_i)
 */
  inline tensor sinh(const tensor &x) { return dynet::sinh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic cosine
 * \details Elementwise calculation of the hyperbolic cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to cosh(x_i)
 */
  inline tensor cosh(const tensor &x) { return dynet::cosh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic tangent
 * \details Elementwise calculation of the hyperbolic tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to tanh(x_i)
 */
  inline tensor tanh(const tensor &x) { return dynet::tanh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic sine
 * \details Elementwise calculation of the inverse hyperbolic sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to asinh(x_i)
 */
  inline tensor asinh(const tensor &x) { return dynet::asinh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic cosine
 * \details Elementwise calculation of the inverse hyperbolic cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to acosh(x_i)
 */
  inline tensor acosh(const tensor &x) { return dynet::acosh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic tangent
 * \details Elementwise calculation of the inverse hyperbolic tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to atanh(x_i)
 */
  inline tensor atanh(const tensor &x) { return dynet::atanh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Natural exponent
 * \details Calculate elementwise y_i = e^{x_i}
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to e^{x_i}
 */
  inline tensor exp(const tensor &x) { return dynet::exp(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Square
 * \details Calculate elementwise y_i = x_i^2
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^2
 */
  inline tensor square(const tensor &x) { return dynet::square(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Cube
 * \details Calculate elementwise y_i = x_i^3
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^3
 */
  inline tensor cube(const tensor &x) { return dynet::cube(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Log sigmoid
 * \details Calculate elementwise \f$y_i = \ln(\frac{1}{1+e^{-x_i}})\f$
 * This is more numerically stable than `log(logistic(x))`
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$y_i = \ln(\frac{1}{1+e^{-x_i}})\f$
 */
  inline tensor log_sigmoid(const tensor &x) { return dynet::log_sigmoid(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Log gamma
 * \details Calculate elementwise y_i = ln(gamma(x_i))
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(gamma(x_i))
 */
  inline tensor lgamma(const tensor &x) { return dynet::lgamma(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Logarithm
 * \details Calculate the elementwise natural logarithm y_i = ln(x_i)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(x_i)
 */
  inline tensor log(const tensor &x) { return dynet::log(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Logistic sigmoid function
 * \details Calculate elementwise y_i = 1/(1+e^{-x_i})
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to y_i = 1/(1+e^{-x_i})
 */
  inline tensor logistic(const tensor &x) { return dynet::logistic(x); }
  inline tensor sigmoid(const tensor &x) { return dynet::logistic(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Rectifier
 * \details Calculate elementwise the recitifer (ReLU) function y_i = max(x_i,0)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to max(x_i,0)
 */
  inline tensor rectify(const tensor &x) { return dynet::rectify(x); }


/**
 * \ingroup arithmeticoperations
 * \brief Exponential Linear Unit
 * \details Calculate elementwise the function
 *
 * \f$
 * y_i = \left\{\begin{array}{lr}
 *            x_i, & \text{if } x>0\\
 *            \alpha\times(e^{x_i} - 1), & \text{if }x\leqslant 0\\
 *          \end{array}\right.
 * \f$
 *
 * Reference: [Clevert et al., 2015](https://arxiv.org/abs/1511.07289v5)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\text{ELU}(x_i, \alpha)\f$
 */
  inline tensor elu(const tensor &x, float alpha = 1.f) { return dynet::elu(x, alpha); }

/**
 * \ingroup arithmeticoperations
 * \brief Scaled Exponential Linear Unit (SELU)
 * \details Calculate elementwise the function
 *
 * \f$
 * y_i = \lambda\times\left\{\begin{array}{lr}
 *            x_i, & \text{if } x>0\\
 *            \alpha\times(e^{x_i} - 1), & \text{if }x\leqslant 0\\
 *          \end{array}\right.
 * \f$
 *
 * With
 * \f$
 * \begin{split}
 * \lambda &=\texttt{1.0507009873554804934193349852946}\\
 * \alpha &=\texttt{1.6732632423543772848170429916717}\\
 * \end{split}
 * \f$
 *
 * Reference: [Klambaouer et al., 2017](https://arxiv.org/abs/1706.02515)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\text{SELU}(x_i)\f$
 */
  inline tensor selu(const tensor &x) { return dynet::selu(x); }

/**
 * \ingroup arithmeticoperations
 * \brief SILU / SiL / Swish
 * \details Calculate elementwise y_i = x_i / (1 + e^{-beta * x_i})
 *
 * Reference: [Hendrycks and Gimpel, 2016](https://openreview.net/pdf?id=Bk0MRI5lg),
 * [Elfwing et al, 2017](https://arxiv.org/pdf/1702.03118.pdf), and
 * [Ramachandran et al., 2017](https://arxiv.org/pdf/1710.05941)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to y_i = x_i / (1 + e^{-beta * x_i})
 */
  inline tensor silu(const tensor &x, float beta = 1.f) { return dynet::silu(x, beta); }

/**
 * \ingroup arithmeticoperations
 * \brief Soft Sign
 * \details Calculate elementwise the softsign function y_i = x_i/(1+|x_i|)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i/(1+|x_i|)
 */
  inline tensor softsign(const tensor &x) { return dynet::softsign(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Power function
 * \details Calculate an output where the ith element is equal to x_i^y
 *
 * \param x The input expression
 * \param y The exponent expression(scalar expression)
 *
 * \return An expression where the ith element is equal to x_i^y
 */
  inline tensor pow(const tensor &x, const tensor &y) { return dynet::pow(x, y); }


  inline tensor gelu(const tensor& x) {
    return cmult(0.5*x, (1.0+tanh(0.79788456*(x + 0.044715*cube(x)))));
  }

/**
 * \ingroup arithmeticoperations
 * \brief Minimum
 * \details Calculate an output where the ith element is min(x_i,y_i)
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to min(x_i,y_i)
 */
  inline tensor min(const tensor &x, const tensor &y) { return dynet::min(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Maximum
 * \details Calculate an output where the ith element is max(x_i,y_i)
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to max(x_i,y_i)
 */
  inline tensor max(const tensor &x, const tensor &y) { return dynet::max(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Max
 * \details This performs an elementwise max over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to max(xs[0][i], xs[1][i], ...)
 */
  inline tensor max(const std::vector<tensor> &xs) {
    if (xs.empty()) { throw std::runtime_error("cannot perform max on empty list"); }
    auto extra_dim = xs[0].dim().nd;
    return dynet::max_dim(dynet::concatenate(tensor::vector_cast_to_base(xs), extra_dim), extra_dim);
  }


/**
 * \ingroup arithmeticoperations
 * \brief Dot Product
 * \details Calculate the dot product sum_i x_i*y_i
 *
 * \param x The input expression
 * \param y The input expression
 *
 * \return An expression equal to the dot product
 */
  inline tensor dot_product(const tensor &x, const tensor &y) { return dynet::dot_product(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Circular convolution
 * \details Calculate the circular convolution
 *
 * \param x The input expression
 * \param y The input expression
 *
 * \return An expression equal to the circular convolution
 */
  inline tensor circ_conv(const tensor &u, const tensor &v) { return dynet::circ_conv(u, v); }

/**
 * \ingroup arithmeticoperations
 * \brief Circular correlation
 * \details Calculate the circular correlation
 *
 * \param x The input expression
 * \param y The input expression
 *
 * \return An expression equal to the circular correlation
 */
  inline tensor circ_corr(const tensor &u, const tensor &v) { return dynet::circ_corr(u, v); }

/**
 * \ingroup arithmeticoperations
 * \brief Componentwise multiply
 * \details Multiply two expressions component-wise, broadcasting dimensions if necessary as follows:
 *          - When number of dimensions differ, we add dimensions of size 1 to make the number of dimensions match
 *          - Now, every dimensions is required to have matching size, or one of the dimensions must equal 1 (in which case it will be broadcasted)
 *          - In the same way, the batch dimension must match, or equal 1 in which case it will be broadcasted
 *          - The resulting tensor's dimensionality is thus determined as the max of both inputs at every position
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to x_i*y_i
 */
  inline tensor cmult(const tensor &x, const tensor &y) { return dynet::cmult(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Componentwise division
 * \details Divide an expressions component-wise by another, broadcasting dimensions (currently only of the second expression!) if necessary as follows:
 *          - When number of dimensions differ, we add dimensions of size 1 to make the number of dimensions match
 *          - Now, every dimensions is required to have matching size, or the dim size of the right expression must equal 1 (in which case it will be broadcasted)
 *          - In the same way, the batch sizes must match, or the batch size of the right expression must equal 1 in which case it will be broadcasted
 *          - The resulting tensor's dimensionality is thus determined as the max of both inputs at every position
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to x_i/y_i
 */
  inline tensor cdiv(const tensor &x, const tensor &y) { return dynet::cdiv(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Columnwise addition
 * \details Add vector "bias" to each column of matrix "x"
 *
 * \param x An MxN matrix
 * \param bias A length M vector
 *
 * \return An expression where bias is added to each column of x
 */
  inline tensor colwise_add(const tensor &x, const tensor &bias) { return dynet::colwise_add(x, bias); }

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

/**
 * \ingroup lossoperations
 * \brief Softmax
 * \details The softmax function normalizes each column to ensure that all
 *          values are between 0 and 1 and add to one by applying
 *          \f$\frac{e^{x_i}}{\sum_j e^{x_j}}\f$.
 *
 * \param x A vector or matrix
 * \param d dimension to normalize over (default: 0)
 *
 * \return A vector or matrix after calculating the softmax
 */
  inline tensor softmax(const tensor &x, unsigned d = 0) { return dynet::softmax(x, d); }

/**
 * \ingroup lossoperations
 * \brief Log softmax
 * \details The log softmax function normalizes each column to ensure that all
 *          values are between 0 and 1 and add to one by applying
 *          \f$\frac{e^{x_i}}{\sum_j e^{x_j}}\f$, then taking the log
 *
 * \param x A vector or matrix
 *
 * \return A vector or matrix after calculating the log softmax
 */
  inline tensor log_softmax(const tensor &x) { return dynet::log_softmax(x); }

/**
 * \ingroup lossoperations
 * \brief Restricted log softmax
 * \details The log softmax function calculated over only a subset of the vector elements. The
 *          elements to be included are set by the ``restriction`` variable. All elements not
 *          included in ``restriction`` are set to negative infinity.
 *
 * \param x A vector over which to calculate the softmax
 * \param restriction The elements over which to calculate the softmax
 *
 * \return A vector with the log softmax over the specified elements
 */
  inline tensor log_softmax(const tensor &x, const std::vector<unsigned> &restriction) {
    return dynet::log_softmax(x, restriction);
  }

/**
 * \ingroup lossoperations
 * \brief Log, sum, exp by dimension
 * \details The "logsumexp" function calculated over a particular dimension
 *   \f$ln(\sum_i e^{xs_i})\f$, used in adding probabilities in the log domain.
 *
 * \param x Expression with respect to which to calculate the logsumexp.
 * \param d The dimension along which to do the logsumexp.
 *
 * \return The result.
 */
  inline tensor logsumexp_dim(const tensor &x, unsigned d) { return dynet::logsumexp_dim(x, d); }

/**
 * \ingroup lossoperations
 * \brief Log, sum, exp
 * \details The elementwise "logsumexp" function that calculates
 *   \f$ln(\sum_i e^{xs_i})\f$, used in adding probabilities in the log domain.
 *
 * \param xs Expressions with respect to which to calculate the logsumexp.
 *
 * \return The result.
 */
  inline tensor logsumexp(const std::vector<tensor> &xs) { return dynet::logsumexp(tensor::vector_cast_to_base(xs)); }


/**
 * \ingroup lossoperations
 * \brief Negative softmax log likelihood
 * \details This function takes in a vector of scores ``x``, and performs a log softmax, takes
 *          the negative, and selects the likelihood corresponding to the element ``v``. This is
 *          perhaps the most standard loss function for training neural networks to predict
 *          one out of a set of elements.
 *
 * \param x A vector of scores
 * \param v The element with which to calculate the loss
 *
 * \return The negative log likelihood of element ``v`` after taking the softmax
 */
  inline tensor pickneglogsoftmax(const tensor &x, unsigned v) { return dynet::pickneglogsoftmax(x, v); }



/**
 * \ingroup lossoperations
 * \brief Hinge loss
 * \details This expression calculates the hinge loss, formally expressed as:
 *          \f$ \text{hinge}(x,index,m) = \sum_{i \ne index} \max(0, m-x[index]+x[i]). \f$
 *
 * \param x A vector of scores
 * \param index The index of the correct candidate
 * \param m The margin
 *
 * \return The hinge loss of candidate ``index`` with respect to margin ``m``
 */
  inline tensor hinge(const tensor &x, unsigned index, float m = 1.0) { return dynet::hinge(x, index, m); }


/**
 * \ingroup lossoperations
 * \brief Dimensionwise hinge loss
 * \details This expression calculates the hinge loss over a particular dimension ``d``.
 *
 * \param x A matrix of scores
 * \param indices The indices of the correct candidate (equal in length to the
 *                dimension not specified by "d")
 * \param d The dimension over which to calculate the loss (0 or 1)
 * \param m The margin
 *
 * \return A vector of hinge losses for each index in ``indices``.
 */
  inline tensor hinge_dim(const tensor &x, const std::vector<unsigned> &indices, unsigned d = 0,
                          float m = 1.0) { return dynet::hinge_dim(x, indices, d, m); }


/**
 * \ingroup lossoperations
 * \brief Batched dimensionwise hinge loss
 * \details The same as dimensionwise hinge loss, but for the case where ``x`` is a mini-batched tensor
 *          with ``indices.size()`` batch elements.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param indices The indices of the correct candidates for each batch element
 * \param d The dimension over which to calculate the loss (0 or 1)
 * \param m The margin
 *
 * \return A vector of hinge losses for each mini-batch
 */
  inline tensor hinge_dim(const tensor &x, const std::vector<std::vector<unsigned> > &indices, unsigned d = 0,
                          float m = 1.0) { return dynet::hinge_dim(x, indices, d, m); }


/**
 * \ingroup lossoperations
 * \brief Sparsemax
 * \details The sparsemax function (Martins et al. 2016), which is similar to softmax,
 *          but induces sparse solutions where most of the vector elements are zero.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 *
 * \return The sparsemax of the scores
 */
  inline tensor sparsemax(const tensor &x) { return dynet::sparsemax(x); }

/**
 * \ingroup lossoperations
 * \brief Sparsemax loss
 * \details The sparsemax loss function (Martins et al. 2016), which is similar to
 *          softmax loss, but induces sparse solutions where most of the vector
 *          elements are zero. It has a gradient similar to the sparsemax function
 *          and thus is useful for optimizing when the sparsemax will be used at
 *          test time.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 * \param target_support The target correct labels.
 *
 * \return The sparsemax loss of the labels
 */
  inline tensor sparsemax_loss(const tensor &x, const std::vector<unsigned> &target_support) {
    return dynet::sparsemax_loss(x, target_support);
  }


/**
 * \ingroup lossoperations
 * \brief Constrained softmax
 * \details The constrained softmax function.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 * \param y A vector of upper bound constraints on probabilities
 *
 * \return The constrained softmax of the scores.
 */
  inline tensor constrained_softmax(const tensor &x, const tensor &y) {
    return dynet::constrained_softmax(x, y);
  }

/**
 * \ingroup lossoperations
 * \brief Squared norm
 * \details The squared L2 norm of the values of x: \f$\sum_i x_i^2\f$.
 *
 * \param x A vector of values
 *
 * \return The squared L2 norm
 */
  inline tensor squared_norm(const tensor &x) { return dynet::squared_norm(x); }

/**
 * \ingroup lossoperations
 * \brief L2 norm
 * \details The L2 norm of the values of x: \f$\sqrt{\sum_i x_i^2}\f$.
 *
 * \param x A vector of values
 *
 * \return The L2 norm
 */
  inline tensor l2_norm(const tensor &x) { return dynet::l2_norm(x); }

/**
 * \ingroup lossoperations
 * \brief Squared distance
 * \details The squared distance between values of ``x`` and ``y``: \f$\sum_i (x_i-y_i)^2\f$.
 *
 * \param x A vector of values
 * \param y Another vector of values
 *
 * \return The squared distance
 */
  inline tensor squared_distance(const tensor &x, const tensor &y) {
    return dynet::squared_distance(x, y);
  }

/**
 * \ingroup lossoperations
 * \brief L1 distance
 * \details The L1 distance between values of ``x`` and ``y``: \f$\sum_i |x_i-y_i|\f$.
 *
 * \param x A vector of values
 * \param y Another vector of values
 *
 * \return The squared distance
 */
  inline tensor l1_distance(const tensor &x, const tensor &y) { return dynet::l1_distance(x, y); }

/**
 * \ingroup lossoperations
 * \brief Huber distance
 * \details The huber distance between values of ``x`` and ``y`` parameterized
 *    by ``c,`` \f$\sum_i L_c(x_i, y_i)\f$ where:
 *
 *    \f$
 *      L_c(x, y) = \begin{cases}
 *        \frac{1}{2}(y - x)^2                   & \textrm{for } |y - f(x)| \le c, \\
 *        c\, |y - f(x)| - \frac{1}{2}c^2 & \textrm{otherwise.}
 *      \end{cases}
 *    \f$
 *
 * \param x A vector of values
 * \param y Another vector of values
 * \param c The parameter of the huber distance parameterizing the cuttoff
 *
 * \return The huber distance
 */
  inline tensor huber_distance(const tensor &x, const tensor &y, float c = 1.345f) {
    return dynet::huber_distance(x, y, c);
  }

/**
 * \ingroup lossoperations
 * \brief Binary log loss
 * \details The log loss of a binary decision according to the sigmoid
 *          sigmoid function \f$- \sum_i (y_i * ln(x_i) + (1-y_i) * ln(1-x_i)) \f$
 *
 * \param x A vector of values
 * \param y A vector of true answers
 *
 * \return The log loss of the sigmoid function
 */
  inline tensor binary_log_loss(const tensor &x, const tensor &y) { return dynet::binary_log_loss(x, y); }

/**
 * \ingroup lossoperations
 * \brief Pairwise rank loss
 * \details A margin-based loss, where every margin violation for each pair of
 *          values is penalized: \f$\sum_i max(m - x_i + y_i, 0)\f$
 *
 * \param x A vector of values
 * \param y A vector of true answers
 * \param m The margin
 *
 * \return The pairwise rank loss
 */
  inline tensor pairwise_rank_loss(const tensor &x, const tensor &y, float m = 1.0) {
    return dynet::pairwise_rank_loss(x, y, m);
  }

/**
 * \ingroup lossoperations
 * \brief Poisson loss
 * \details The negative log probability of ``y`` according to a Poisson
 *          distribution with parameter ``x``. Useful in Poisson regression
 *          where, we try to predict the parameters of a Possion distribution
 *          to maximize the probability of data ``y``.
 *
 * \param x The parameter of the Poisson distribution.
 * \param y The target value
 *
 * \return The Poisson loss
 */
  inline tensor poisson_loss(const tensor &x, unsigned y) { return dynet::poisson_loss(x, y); }


////////////////////////////////////////////////
// Flow operations                            //
////////////////////////////////////////////////

/**
 * \ingroup flowoperations
 * \brief Prevent backprop
 * \details This node has no effect on the forward pass, but prevents gradients from
 *          flowing backward during the backward pass. This is useful when there's
 *          a subgraph for which you don't want loss passed back to the parameters.
 *
 * \param x The input expression
 *
 * \return The new expression
 */
  inline tensor nobackprop(const tensor &x) { return dynet::nobackprop(x); }

/**
 * \ingroup flowoperations
 * \brief Flip gradient
 * \details This node has no effect on the forward pass, but inverts the gradient on backprop.
 *          This operation is widely used in adversarial networks.
 *
 * \param x The input expression
 *
 * \return An output expression containing the same as input (only effects the backprop process)
 */
  inline tensor flip_gradient(const tensor &x) { return dynet::flip_gradient(x); }

/**
 * \ingroup flowoperations
 * \brief Scale gradient by constant
 * \details This node has no effect on the forward pass, but scales the gradient by lambda
 *          on backprop
 *
 * \param x The input expression
 *
 * \return An output expression containing the same as input (only effects the backprop process)
 */
  inline tensor scale_gradient(const tensor &x, float lambd = 1.0f) {
    return dynet::scale_gradient(x, lambd);
  }


/**
 * \ingroup flowoperations
 * \brief Strided select in multiple dimensions
 * \details Select a range and/or stride of elements from an expression.
 *
 * \param x The input expression
 * \param strides List of strides for each dimension, must be >= 1. Dimensions not included default to 1.
 * \param from    List of 0-based offsets (inclusive) for each dimension, must be >= 0. Dimensions not included default to 0.
 * \param to      List of highest 0-based index to select (exclusive) for each dimension, must be >= 0. Dimensions not included default to the corresponding dim size.
 *
 * \return The value of x[from[0]:to[0]:strides[0],..] (as it would be in numpy syntax)
 */
  inline tensor
  strided_select(const tensor &x, const std::vector<int> &strides, const std::vector<int> &from = {},
                 const std::vector<int> &to = {}) { return dynet::strided_select(x, strides); }


/**
 * \ingroup flowoperations
 * \brief Concatenate columns
 * \details Perform a concatenation of the columns in multiple expressions.
 *          All expressions must have the same number of rows.
 *
 * \param xs The input expressions
 *
 * \return The expression with the columns concatenated
 */
  inline tensor concatenate_cols(const std::vector<tensor> &xs) {
    return dynet::concatenate_cols(tensor::vector_cast_to_base(xs));
  }

/**
 * \ingroup flowoperations
 * \brief Concatenate
 * \details Perform a concatenation of multiple expressions along
 *          a particular dimension.
 *          All expressions must have the same dimensions except for
 *          the dimension to be concatenated (rows by default).
 *
 * \param xs The input expressions
 * \param d The dimension along which to perform concatenation
 *
 * \return The expression with the specified dimension concatenated
 */
  inline tensor
  concatenate(const std::initializer_list<tensor> &xs, unsigned d = 0) {
    return dynet::concatenate(tensor::vector_cast_to_base(xs), d);
  }

  inline tensor concatenate(const std::vector<tensor> &xs, unsigned d = 0) {
    return dynet::concatenate(tensor::vector_cast_to_base(xs), d);
  }

/**
 * \ingroup flowoperations
 * \brief Max out through a dimension
 * \details Select out a element/row/column/sub-tensor from an expression,
 *          with maximum value along a given dimension.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param d The dimension along which to choose the element
 *
 * \return An expression of sub-tensor with max value along dimension d
 */
  inline tensor max_dim(const tensor &x, unsigned d = 0) {
    auto dim = x.dim();

    // use maxpooling2d under the hood when possible
    // because maxpooling2d is faster than max_dim during backprop
    if(dim.ndims() == 2) {
      if(d==0) {
        return dynet::reshape(dynet::maxpooling2d(x, {dim[0], 1}, {dim[0], 1}), {dim[1]});
      }
      else {
        return dynet::reshape(dynet::maxpooling2d(x, {1, dim[1]}, {1, dim[1]}), {dim[0]});
      }
    }

    return dynet::max_dim(x, d);
  }

/**
 * \ingroup flowoperations
 * \brief Min out through a dimension
 * \details Select out a element/row/column/sub-tensor from an expression,
 *          with minimum value along a given dimension.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param d The dimension along which to choose the element
 *
 * \return An expression of sub-tensor with min value along dimension d
 */
  inline tensor min_dim(const tensor &x, unsigned d = 0) { return dynet::min_dim(x, d); }


////////////////////////////////////////////////
// Noise operations                           //
////////////////////////////////////////////////

/**
 * \ingroup noiseoperations
 * \brief Gaussian noise
 * \details Add gaussian noise to an expression.
 *
 * \param x The input expression
 * \param stddev The standard deviation of the gaussian
 *
 * \return The noised expression
 */
  inline tensor noise(const tensor &x, float stddev) { return dynet::noise(x, stddev); }

/**
 * \ingroup noiseoperations
 * \brief Dropout
 * \details
 *   With a fixed probability, drop out (set to zero) nodes in the input
 *   expression, and **scale** the remaining nodes by 1/p. Note that there are
 *   [two kinds of dropout](http://cs231n.github.io/neural-networks-2/#reg):
 *   - *Regular dropout:* where we perform dropout at training time and then\n
 *     scale outputs by p at test time.
 *   - *Inverted dropout:* where we perform dropout and scaling at training\n
 *     time, and do not need to do anything at test time.
 *   DyNet implements the latter, so you only need to apply dropout at training
 *   time, and do not need to perform scaling and test time.
 *
 * \param x The input expression
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
  inline tensor dropout(const tensor &x, float p) { return dynet::dropout(x, p); }

/**
 * \ingroup noiseoperations
 * \brief Dropout along a specific dimension
 * \details Identical to the dropout operation except the dropout mask is the same across one dimension. Use this if you want to drop columns or lines in a matrix for example
 *
 * For now this only supports tensors of order <= 3
 *
 * \param x The input expression
 * \param d The dimension along which to drop
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
  inline tensor dropout_dim(const tensor &x, unsigned d, float p) { return dynet::dropout_dim(x, d, p); }

/**
 * \ingroup noiseoperations
 * \brief Block dropout
 * \details Identical to the dropout operation, but either drops out *all*
 *          or *no* values in the expression, as opposed to making a decision
 *          about each value individually.
 *
 * \param x The input expression
 * \param p The block dropout probability
 *
 * \return The block dropout expression
 */
  inline tensor block_dropout(const tensor &x, float p) { return dynet::block_dropout(x, p); }


////////////////////////////////////////////////
// Tensor operations                          //
////////////////////////////////////////////////

/**
 * \ingroup tensoroperations
 * \brief Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor
 * \details The resulting tensor \f$z\f$ has coordinates \f$z_ij = \sum_k x_{ijk} y_k\f$
 *
 * \param x Rank 3 tensor
 * \param y Vector
 *
 * \return Matrix
 */
  inline tensor contract3d_1d(const tensor &x, const tensor &y) { return dynet::contract3d_1d(x, y); }
// z_i = x_ijk * y_k * z_j (+ b_i)
/**
 * \ingroup tensoroperations
 * \brief Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor
 * \details This is the equivalent of calling `contract3d_1d` and then performing a matrix vector multiplication.
 *
 * The resulting tensor \f$t\f$ has coordinates \f$t_i = \sum_{j,k} x_{ijk} y_k z_j\f$
 *
 * \param x Rank 3 tensor
 * \param y Vector
 * \param z Vector
 * \return Vector
 */
  inline tensor contract3d_1d_1d(const tensor &x, const tensor &y, const tensor &z) {
    return dynet::contract3d_1d_1d(x, y, z);
  }

/**
 * \ingroup tensoroperations
 * \brief Same as `contract3d_1d_1d` with an additional bias parameter
 * \details This is the equivalent of calling `contract3d_1d` and then performing an affine transform.
 *
 * The resulting tensor \f$t\f$ has coordinates \f$t_i = b_i + \sum_{j,k} x_{ijk} y_k z_j\f$
 *
 * \param x Rank 3 tensor
 * \param y Vector
 * \param z Vector
 * \param b Bias vector
 * \return Vector
 */
  inline tensor contract3d_1d_1d(const tensor &x, const tensor &y, const tensor &z,
                                 const tensor &b) { return dynet::contract3d_1d_1d(x, y, z, b); }
// z_ij = x_ijk * y_k + b_ij
/**
 * \ingroup tensoroperations
 * \brief Same as `contract3d_1d` with an additional bias parameter
 * \details The resulting tensor \f$z\f$ has coordinates \f$z_{ij} = b_{ij}+\sum_k x_{ijk} y_k\f$
 *
 * \param x Rank 3 tensor
 * \param y Vector
 * \param b Bias matrix
 * \return Matrix
 */
  inline tensor contract3d_1d(const tensor &x, const tensor &y, const tensor &b) {
    return dynet::contract3d_1d(x, y, b);
  }


////////////////////////////////////////////////
// Linear algebra operations                  //
////////////////////////////////////////////////

/**
 * \ingroup linalgoperations
 * \brief Matrix Inverse
 * \details Takes the inverse of a matrix (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *          Note that back-propagating through an inverted matrix can also be the
 *          source of stability problems sometimes.
 *
 * \param x A square matrix
 *
 * \return The inverse of the matrix
 */
  inline tensor inverse(const tensor &x) { return dynet::inverse(x); }

/**
 * \ingroup linalgoperations
 * \brief Log determinant
 * \details Takes the log of the determinant of a matrix.
 *          (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *
 * \param x A square matrix
 *
 * \return The log of its determinant
 */
  inline tensor logdet(const tensor &x) { return dynet::logdet(x); }

/**
 * \ingroup linalgoperations
 * \brief Trace of Matrix Product
 * \details Takes the trace of the product of matrices.
 *          (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *
 * \param x1 A matrix
 * \param x2 Another matrix
 *
 * \return trace(x1 * x2)
 */
  inline tensor trace_of_product(const tensor &x, const tensor &y) {
    return dynet::trace_of_product(x, y);
  }

////////////////////////////////////////////////
// Normalization operations                   //
////////////////////////////////////////////////

/**
 * \ingroup normoperations
 * \brief Layer normalization
 * \details Performs layer normalization :
 *
 * \f$
 * \begin{split}
 *    \mu &= \frac 1 n \sum_{i=1}^n x_i\\
 *    \sigma &= \sqrt{\frac 1 n \sum_{i=1}^n (x_i-\mu)^2}\\
 *    y&=\frac {\boldsymbol{g}} \sigma \circ (\boldsymbol{x}-\mu) + \boldsymbol{b}\\
 * \end{split}
 * \f$
 *
 * Reference : [Ba et al., 2016](http://arxiv.org/abs/1607.06450)
 *
 * \param x Input expression
 * \param g Gain
 * \param b Bias
 * \return An expression of the same dimension as `x`
 */
  inline tensor
  layer_norm(const tensor &x, const tensor &g, const tensor &b) { return dynet::layer_norm(x, g, b); }


  /**
   * find the index of the largest value
   * \param logits must be of dim{n}
   * \return
   */
  inline unsigned argmax_index(const tensor &logits) {
    auto logits_value = as_vector(dyana::_cg().incremental_forward(logits));

    float max_value = logits_value[0];
    unsigned max_index = 0;
    for (unsigned i = 1; i < logits_value.size(); ++i) {
      float val = logits_value[i];
      if (val > max_value) {
        max_value = val;
        max_index = i;
      }
    }
    return max_index;
  }
}

#endif //DYANA_OPERATIONS_HPP
