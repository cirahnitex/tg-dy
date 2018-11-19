//
// Created by YAN Yuchen on 11/13/2018.
//

#ifndef DYNET_WRAPPER_DY_OPERATIONS_HPP
#define DYNET_WRAPPER_DY_OPERATIONS_HPP

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "dy_common.hpp"

namespace tg {
  namespace dy {

    typedef dynet::Dim Dim;

    inline Expression zeros(const Dim& d) {return dynet::zeros(cg(), d);}
    inline Expression ones(const Dim& d) {return dynet::ones(cg(), d);}

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
    inline Expression affine_transform(const std::initializer_list<Expression> &xs) {
      return dynet::affine_transform(Expression::vector_cast_to_base(xs));
    }

    inline Expression affine_transform(const std::vector<Expression> &xs) { return dynet::affine_transform(Expression::vector_cast_to_base(xs)); }

/**
 * \ingroup arithmeticoperations
 * \brief Sum
 * \details This performs an elementwise sum over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to xs[0][i] + xs[1][i] + ...
 */
    inline Expression sum(const std::vector<Expression> &xs) { return dynet::sum(Expression::vector_cast_to_base(xs)); }


/**
 * \ingroup arithmeticoperations
 * \brief Sum all elements
 * \details Sum all the elements in an expression.
 *
 * \param x The input expression
 *
 * \return The sum of all of its elements
 */
    inline Expression sum_elems(const Expression &x) { return dynet::sum_elems(x); }

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
    inline Expression moment_elems(const Expression &x, unsigned r) { return dynet::moment_elems(x, r); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
    inline Expression mean_elems(const Expression &x) { return dynet::mean_elems(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute Standard deviation over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
    inline Expression std_elems(const Expression &x) { return dynet::std_elems(x); }

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
    inline Expression sum_batches(const Expression &x) { return dynet::sum_batches(x); }

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
    inline Expression moment_batches(const Expression &x, unsigned r) { return dynet::moment_batches(x, r); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension
 *
 * \param x The input mini-batched expression
 *
 * \return An expression with a single batch
 */
    inline Expression mean_batches(const Expression &x) { return dynet::mean_batches(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Compute standard deviation over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
    inline Expression std_batches(const Expression &x) { return dynet::std_batches(x); }

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
    inline Expression sum_dim(const Expression &x, const std::vector<unsigned> &dims, bool b = false) {
      return dynet::sum_dim(x, dims, b);
    }

// These are deprecated but kept for backward compatibility
    inline Expression sum_rows(const Expression &x) { return dynet::sum_rows(x); }

    inline Expression sum_cols(const Expression &x) { return dynet::sum_cols(x); }

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
    inline Expression cumsum(const Expression &x, unsigned d) { return dynet::cumsum(x, d); }

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
    inline Expression moment_dim(const Expression &x, const std::vector<unsigned> &dims, unsigned r, bool b = false,
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
    inline Expression mean_dim(const Expression &x, const std::vector<unsigned> &dims, bool b = false,
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
    inline Expression std_dim(const Expression &x, const std::vector<unsigned> &dims, bool b = false,
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
    inline Expression average(const std::vector<Expression> &xs) { return dynet::average(Expression::vector_cast_to_base(xs)); }

/**
 * \ingroup arithmeticoperations
 * \brief Square root
 * \details Elementwise square root.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\sqrt(x_i)\f$
 */
    inline Expression sqrt(const Expression &x) { return dynet::sqrt(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Absolute value
 * \details Elementwise absolute value.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\vert x_i\vert\f$
 */
    inline Expression abs(const Expression &x) { return dynet::abs(x); }


/**
 * \ingroup arithmeticoperations
 * \brief Gaussian error function
 * \details Elementwise calculation of the Gaussian error function
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to erf(x_i)
 */
    inline Expression erf(const Expression &x) { return dynet::erf(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse sine
 * \details Elementwise calculation of the inverse sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to asin(x_i)
 */
    inline Expression asin(const Expression &x) { return dynet::asin(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse cosine
 * \details Elementwise calculation of the inverse cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to acos(x_i)
 */
    inline Expression acos(const Expression &x) { return dynet::acos(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse tangent
 * \details Elementwise calculation of the inverse tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to atan(x_i)
 */
    inline Expression atan(const Expression &x) { return dynet::atan(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Sine
 * \details Elementwise calculation of the sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to sin(x_i)
 */
    inline Expression sin(const Expression &x) { return dynet::sin(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Cosine
 * \details Elementwise calculation of the cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to cos(x_i)
 */
    inline Expression cos(const Expression &x) { return dynet::cos(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Tangent
 * \details Elementwise calculation of the tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to tan(x_i)
 */
    inline Expression tan(const Expression &x) { return dynet::tan(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic sine
 * \details Elementwise calculation of the hyperbolic sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to sinh(x_i)
 */
    inline Expression sinh(const Expression &x) { return dynet::sinh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic cosine
 * \details Elementwise calculation of the hyperbolic cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to cosh(x_i)
 */
    inline Expression cosh(const Expression &x) { return dynet::cosh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic tangent
 * \details Elementwise calculation of the hyperbolic tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to tanh(x_i)
 */
    inline Expression tanh(const Expression &x) { return dynet::tanh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic sine
 * \details Elementwise calculation of the inverse hyperbolic sine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to asinh(x_i)
 */
    inline Expression asinh(const Expression &x) { return dynet::asinh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic cosine
 * \details Elementwise calculation of the inverse hyperbolic cosine
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to acosh(x_i)
 */
    inline Expression acosh(const Expression &x) { return dynet::acosh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Inverse hyperbolic tangent
 * \details Elementwise calculation of the inverse hyperbolic tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to atanh(x_i)
 */
    inline Expression atanh(const Expression &x) { return dynet::atanh(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Natural exponent
 * \details Calculate elementwise y_i = e^{x_i}
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to e^{x_i}
 */
    inline Expression exp(const Expression &x) { return dynet::exp(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Square
 * \details Calculate elementwise y_i = x_i^2
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^2
 */
    inline Expression square(const Expression &x) { return dynet::square(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Cube
 * \details Calculate elementwise y_i = x_i^3
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^3
 */
    inline Expression cube(const Expression &x) { return dynet::cube(x); }

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
    inline Expression log_sigmoid(const Expression &x) { return dynet::log_sigmoid(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Log gamma
 * \details Calculate elementwise y_i = ln(gamma(x_i))
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(gamma(x_i))
 */
    inline Expression lgamma(const Expression &x) { return dynet::lgamma(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Logarithm
 * \details Calculate the elementwise natural logarithm y_i = ln(x_i)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(x_i)
 */
    inline Expression log(const Expression &x) { return dynet::log(x); }

/**
 * \ingroup arithmeticoperations
 * \brief Logistic sigmoid function
 * \details Calculate elementwise y_i = 1/(1+e^{-x_i})
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to y_i = 1/(1+e^{-x_i})
 */
    inline Expression logistic(const Expression &x) { return 0.5 * dynet::tanh(x * 0.5) + 0.5; }

/**
 * \ingroup arithmeticoperations
 * \brief Rectifier
 * \details Calculate elementwise the recitifer (ReLU) function y_i = max(x_i,0)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to max(x_i,0)
 */
    inline Expression rectify(const Expression &x) { return dynet::rectify(x); }


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
    inline Expression elu(const Expression &x, float alpha = 1.f) { return dynet::elu(x, alpha); }

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
    inline Expression selu(const Expression &x) { return dynet::selu(x); }

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
    inline Expression silu(const Expression &x, float beta = 1.f) { return dynet::silu(x, beta); }

/**
 * \ingroup arithmeticoperations
 * \brief Soft Sign
 * \details Calculate elementwise the softsign function y_i = x_i/(1+|x_i|)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i/(1+|x_i|)
 */
    inline Expression softsign(const Expression &x) { return dynet::softsign(x); }

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
    inline Expression pow(const Expression &x, const Expression &y) { return dynet::pow(x, y); }

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
    inline Expression min(const Expression &x, const Expression &y) { return dynet::min(x, y); }

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
    inline Expression max(const Expression &x, const Expression &y) { return dynet::max(x, y); }

/**
 * \ingroup arithmeticoperations
 * \brief Max
 * \details This performs an elementwise max over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to max(xs[0][i], xs[1][i], ...)
 */
    inline Expression max(const std::vector<Expression> &xs) {
      if(xs.empty()) {throw new std::runtime_error("cannot perform max on empty list");}
      auto extra_dim = xs[0].dim().nd;
      return dynet::max_dim(dynet::concatenate(Expression::vector_cast_to_base(xs), extra_dim), extra_dim);
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
    inline Expression dot_product(const Expression &x, const Expression &y) { return dynet::dot_product(x, y); }

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
    inline Expression circ_conv(const Expression &u, const Expression &v) { return dynet::circ_conv(u, v); }

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
    inline Expression circ_corr(const Expression &u, const Expression &v) { return dynet::circ_corr(u, v); }

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
    inline Expression cmult(const Expression &x, const Expression &y) { return dynet::cmult(x, y); }

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
    inline Expression cdiv(const Expression &x, const Expression &y) { return dynet::cdiv(x, y); }

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
    inline Expression colwise_add(const Expression &x, const Expression &bias) { return dynet::colwise_add(x, bias); }

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
    inline Expression softmax(const Expression &x, unsigned d = 0) { return dynet::softmax(x, d); }

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
    inline Expression log_softmax(const Expression &x) { return dynet::log_softmax(x); }

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
    inline Expression log_softmax(const Expression &x, const std::vector<unsigned> &restriction) {
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
    inline Expression logsumexp_dim(const Expression &x, unsigned d) { return dynet::logsumexp_dim(x, d); }

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
    inline Expression logsumexp(const std::vector<Expression> &xs) { return dynet::logsumexp(Expression::vector_cast_to_base(xs)); }


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
    inline Expression pickneglogsoftmax(const Expression &x, unsigned v) { return dynet::pickneglogsoftmax(x, v); }

/**
 * \ingroup lossoperations
 * \brief Modifiable negative softmax log likelihood
 * \details This function calculates the negative log likelihood after the softmax with
 *          respect to index ``*pv``. This computes the same value as the previous function
 *          that passes the index ``v`` by value, but instead passes by pointer so the value
 *          ``*pv`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 *
 * \param x A vector of scores
 * \param pv A pointer to the index of the correct element
 *
 * \return The negative log likelihood of element ``*pv`` after taking the softmax
 */
    inline Expression pickneglogsoftmax(const Expression &x, const unsigned *pv) {
      return dynet::pickneglogsoftmax(x, pv);
    }

/**
 * \ingroup lossoperations
 * \brief Batched negative softmax log likelihood
 * \details This function is similar to standard pickneglogsoftmax, but calculates loss with
 *          respect to multiple batch elements. The input will be a mini-batch of score vectors
 *          where the number of batch elements is equal to the number of indices in ``v``.
 *
 * \param x An expression with vectors of scores over N batch elements
 * \param v A size-N vector indicating the index with respect to all the batch elements
 *
 * \return The negative log likelihoods over all the batch elements
 */
    inline Expression pickneglogsoftmax(const Expression &x, const std::vector<unsigned> &v) {
      return dynet::pickneglogsoftmax(x, v);
    }

/**
 * \ingroup lossoperations
 * \brief Modifiable batched negative softmax log likelihood
 * \details This function is a combination of modifiable pickneglogsoftmax and batched
 *          pickneglogsoftmax: ``pv`` can be modified without re-creating the computation graph.
 *
 * \param x An expression with vectors of scores over N batch elements
 * \param pv A pointer to the indexes
 *
 * \return The negative log likelihoods over all the batch elements
 */
    inline Expression pickneglogsoftmax(const Expression &x, const std::vector<unsigned> *pv) {
      return dynet::pickneglogsoftmax(x, pv);
    }

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
    inline Expression hinge(const Expression &x, unsigned index, float m = 1.0) { return dynet::hinge(x, index, m); }

/**
 * \ingroup lossoperations
 * \brief Modifiable hinge loss
 * \details This function calculates the hinge loss with
 *          with respect to index ``*pindex``. This computes the same value as the previous function
 *          that passes the index ``index`` by value, but instead passes by pointer so the value
 *          ``*pindex`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 *
 * \param x A vector of scores
 * \param pindex A pointer to the index of the correct candidate
 * \param m The margin
 *
 * \return The hinge loss of candidate ``*pindex`` with respect to margin ``m``
 */
    inline Expression hinge(const Expression &x, const unsigned *pindex, float m = 1.0) {
      return dynet::hinge(x, pindex, m);
    }

/**
 * \ingroup lossoperations
 * \brief Batched hinge loss
 * \details The same as hinge loss, but for the case where ``x`` is a mini-batched tensor
 *          with ``indices.size()`` batch elements, and ``indices`` is a vector indicating
 *          the index of each of the correct elements for these elements.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param indices The indices of the correct candidates for each batch element
 * \param m The margin
 *
 * \return The hinge loss of each mini-batch
 */
    inline Expression hinge(const Expression &x, const std::vector<unsigned> &indices, float m = 1.0) {
      return dynet::hinge(x, indices, m);
    }

/**
 * \ingroup lossoperations
 * \brief Batched modifiable hinge loss
 * \details A combination of the previous batched and modifiable hinge loss functions, where
 *          vector ``*pindices`` can be modified.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param pindices Pointer to the indices of the correct candidates for each batch element
 * \param m The margin
 *
 * \return The hinge loss of each mini-batch
 */
    inline Expression hinge(const Expression &x, const std::vector<unsigned> *pindices, float m = 1.0) {
      return dynet::hinge(x, pindices, m);
    }

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
    inline Expression hinge_dim(const Expression &x, const std::vector<unsigned> &indices, unsigned d = 0,
                                float m = 1.0) { return dynet::hinge_dim(x, indices, d, m); }

/**
 * \ingroup lossoperations
 * \brief Modifiable dimensionwise hinge loss
 * \details This function calculates the modifiable version of dimensionwise hinge loss.
 *
 * \param x A vector of scores
 * \param pindex A pointer to the index of the correct candidate
 * \param d The dimension over which to calculate the loss (0 or 1)
 * \param m The margin
 *
 * \return A vector of hinge losses for each index in ``indices``.
 */
    inline Expression hinge_dim(const Expression &x, const std::vector<unsigned> *pindex, unsigned d = 0,
                                float m = 1.0) { return dynet::hinge_dim(x, pindex, d, m); }

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
    inline Expression hinge_dim(const Expression &x, const std::vector<std::vector<unsigned> > &indices, unsigned d = 0,
                                float m = 1.0) { return dynet::hinge_dim(x, indices, d, m); }

/**
 * \ingroup lossoperations
 * \brief Batched modifiable hinge loss
 * \details A combination of the previous batched and modifiable hinge loss functions, where
 *          vector ``*pindices`` can be modified.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param pindices Pointer to the indices of the correct candidates for each batch element
 * \param d The dimension over which to calculate the loss (0 or 1)
 * \param m The margin
 *
 * \return The hinge loss of each mini-batch
 */
    inline Expression
    hinge_dim(const Expression &x, const std::vector<std::vector<unsigned> > *pindices, unsigned d = 0,
              float m = 1.0) { return dynet::hinge_dim(x, pindices, d, m); }

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
    inline Expression sparsemax(const Expression &x) { return dynet::sparsemax(x); }

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
    inline Expression sparsemax_loss(const Expression &x, const std::vector<unsigned> &target_support) {
      return dynet::sparsemax_loss(x, target_support);
    }

/**
 * \ingroup lossoperations
 * \brief Modifiable sparsemax loss
 * \details Similar to the sparsemax loss, but with ptarget_support being a pointer
 *          to a vector, allowing it to be modified without re-creating the compuation
 *          graph.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 * \param ptarget_support A pointer to the target correct labels.
 *
 * \return The sparsemax loss of the labels
 */
    inline Expression sparsemax_loss(const Expression &x, const std::vector<unsigned> *ptarget_support) {
      return dynet::sparsemax_loss(x, ptarget_support);
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
    inline Expression constrained_softmax(const Expression &x, const Expression &y) {
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
    inline Expression squared_norm(const Expression &x) { return dynet::squared_norm(x); }

/**
 * \ingroup lossoperations
 * \brief L2 norm
 * \details The L2 norm of the values of x: \f$\sqrt{\sum_i x_i^2}\f$.
 *
 * \param x A vector of values
 *
 * \return The L2 norm
 */
    inline Expression l2_norm(const Expression &x) { return dynet::l2_norm(x); }

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
    inline Expression squared_distance(const Expression &x, const Expression &y) {
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
    inline Expression l1_distance(const Expression &x, const Expression &y) { return dynet::l1_distance(x, y); }

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
    inline Expression huber_distance(const Expression &x, const Expression &y, float c = 1.345f) {
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
    inline Expression binary_log_loss(const Expression &x, const Expression &y) { return dynet::binary_log_loss(x, y); }

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
    inline Expression pairwise_rank_loss(const Expression &x, const Expression &y, float m = 1.0) {
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
    inline Expression poisson_loss(const Expression &x, unsigned y) { return dynet::poisson_loss(x, y); }

/**
 * \ingroup lossoperations
 * \brief Modifiable Poisson loss
 * \details Similar to Poisson loss, but with the target value passed by
 *          pointer so that it can be modified without re-constructing the
 *          computation graph.
 *
 * \param x The parameter of the Poisson distribution.
 * \param py A pointer to the target value
 *
 * \return The Poisson loss
 */
    inline Expression poisson_loss(const Expression &x, const unsigned *py) { return dynet::poisson_loss(x, py); }

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
    inline Expression nobackprop(const Expression &x) { return dynet::nobackprop(x); }

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
    inline Expression flip_gradient(const Expression &x) { return dynet::flip_gradient(x); }

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
    inline Expression scale_gradient(const Expression &x, float lambd = 1.0f) {
      return dynet::scale_gradient(x, lambd);
    }


/**
 * \ingroup flowoperations
 * \brief Reshape to another size
 * \details This node reshapes a tensor to another size, without changing the
 *          underlying layout of the data. The layout of the data in DyNet is
 *          column-major, so if we have a 3x4 matrix
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
 *        x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
 *        x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *          and transform it into a 2x6 matrix, it will be rearranged as:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{3,1} & x_{2,2} & x_{1,3} & x_{3,3} & x_{2,4} \\
 *        x_{2,1} & x_{1,2} & x_{3,2} & x_{2,3} & x_{1,4} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *         **Note:** This is O(1) for forward, and O(n) for backward.
 *
 * \param x The input expression
 * \param d The new dimensions
 *
 * \return The reshaped expression
 */

    inline Expression reshape(const Expression &x, const Dim &d) { return dynet::reshape(x, d); }

/**
 * \ingroup flowoperations
 * \brief Transpose a matrix
 * \details Transpose a matrix or tensor, or if dims is specified shuffle the
 *          dimensions arbitrarily.
 *          **Note:** This is O(1) if either the row or column dimension is 1,
 *          and O(n) otherwise.
 *
 * \param x The input expression
 * \param dims The dimensions to swap. The ith dimension of the output will be equal
 *          to the dims[i] dimension of the input. dims must have the same number
 *          of dimensions as x.
 *
 * \return The transposed/shuffled expression
 */
    inline Expression
    transpose(const Expression &x, const std::vector<unsigned> &dims = {1, 0}) { return dynet::transpose(x, dims); }

/**
 * \ingroup flowoperations
 * \brief Select rows
 * \details Select a subset of rows of a matrix.
 *
 * \param x The input expression
 * \param rows The rows to extract
 *
 * \return An expression containing the selected rows
 */
    inline Expression select_rows(const Expression &x, const std::vector<unsigned> &rows) {
      return dynet::select_rows(x, rows);
    }

/**
 * \ingroup flowoperations
 * \brief Modifiable select rows
 * \details Select a subset of rows of a matrix, where the elements of prows
 *          can be modified without re-creating the computation graph.
 *
 * \param x The input expression
 * \param prows The rows to extract
 *
 * \return An expression containing the selected rows
 */
    inline Expression select_rows(const Expression &x, const std::vector<unsigned> *prows) {
      return dynet::select_rows(x, prows);
    }

/**
 * \ingroup flowoperations
 * \brief Select columns
 * \details Select a subset of columns of a matrix. select_cols is more
 *          efficient than select_rows since DyNet uses column-major order.
 *
 * \param x The input expression
 * \param columns The columns to extract
 *
 * \return An expression containing the selected columns
 */
    inline Expression select_cols(const Expression &x, const std::vector<unsigned> &cols) {
      return dynet::select_cols(x, cols);
    }

/**
 * \ingroup flowoperations
 * \brief Modifiable select columns
 * \details Select a subset of columns of a matrix, where the elements of pcols
 *          can be modified without re-creating the computation graph.
 *
 * \param x The input expression
 * \param pcolumns The columns to extract
 *
 * \return An expression containing the selected columns
 */
    inline Expression select_cols(const Expression &x, const std::vector<unsigned> *pcols) {
      return dynet::select_cols(x, pcols);
    }

/**
 * \ingroup flowoperations
 * \brief Pick element
 * \details Pick a single element/row/column/sub-tensor from an expression.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param v The index of the element to select
 * \param d The dimension along which to choose the element
 *
 * \return The value of x[v] along dimension d
 */
    inline Expression pick(const Expression &x, unsigned v, unsigned d = 0) { return dynet::pick(x, v, d); }

/**
 * \ingroup flowoperations
 * \brief Batched pick
 * \details Pick elements from multiple batches.
 *
 * \param x The input expression
 * \param v A vector of indicies to choose, one for each batch in the
 *          input expression.
 * \param d The dimension along which to choose the elements
 *
 * \return A mini-batched expression containing the picked elements
 */
    inline Expression pick(const Expression &x, const std::vector<unsigned> &v, unsigned d = 0) {
      return dynet::pick(x, v, d);
    }

/**
 * \ingroup flowoperations
 * \brief Modifiable pick element
 * \details Pick a single element from an expression, where the index is
 *          passed by pointer so we do not need to re-create the computation
 *          graph every time.
 *
 * \param x The input expression
 * \param pv Pointer to the index of the element to select
 * \param d The dimension along which to choose the elements
 *
 * \return The value of x[*pv]
 */
    inline Expression pick(const Expression &x, const unsigned *pv, unsigned d = 0) { return dynet::pick(x, pv, d); }

/**
 * \ingroup flowoperations
 * \brief Modifiable batched pick element
 * \details Pick multiple elements from an input expression, where the indices
 *          are passed by pointer so we do not need to re-create the computation
 *          graph every time.
 *
 * \param x The input expression
 * \param pv A pointer to vector of indicies to choose
 * \param d The dimension along which to choose the elements
 *
 * \return A mini-batched expression containing the picked elements
 */
    inline Expression pick(const Expression &x, const std::vector<unsigned> *pv, unsigned d = 0) {
      return dynet::pick(x, pv, d);
    }

/**
 * \ingroup flowoperations
 * \brief Pick range of elements
 * \details Pick a range of elements from an expression.
 *
 * \param x The input expression
 * \param s The start index
 * \param e The end index
 * \param d The dimension along which to pick
 *
 * \return The value of {x[v],...,x[u]}
 */
    inline Expression
    pick_range(const Expression &x, unsigned s, unsigned e, unsigned d = 0) { return dynet::pick_range(x, s, e, d); }

// DEPRECATED
    inline Expression pickrange(const Expression &x, unsigned s, unsigned e) { return dynet::pickrange(x, s, e); }

/**
 * \ingroup flowoperations
 * \brief (Modifiable) Pick batch element.
 * \details Pick batch element from a batched expression. For a Tensor with 3 batch elements:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1,1} & x_{1,1,2} \\
 *        x_{1,2,1} & x_{1,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * pick_batch_elem(t, 1) will return a Tensor of
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * \param x The input expression
 * \param v The index of the batch element to be picked.
 *
 * \return The expression of picked batch element. The picked element is a tensor
 *         whose `bd` equals to one.
 */
    inline Expression pick_batch_elem(const Expression &x, unsigned v) { return dynet::pick_batch_elem(x, v); }

/**
 * \ingroup flowoperations
 * \brief (Modifiable) Pick batch elements.
 * \details Pick several batch elements from a batched expression. For a Tensor with 3 batch elements:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1,1} & x_{1,1,2} \\
 *        x_{1,2,1} & x_{1,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * pick_batch_elems(t, {1, 2}) will return a Tensor of with 2 batch elements:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * \param x The input expression
 * \param v A vector of indicies of the batch elements to be picked.
 *
 * \return The expression of picked batch elements. The batch elements is a tensor
 *         whose `bd` equals to the size of vector `v`.
 */
    inline Expression
    pick_batch_elems(const Expression &x, const std::vector<unsigned> &v) { return dynet::pick_batch_elems(x, v); }

/**
 * \ingroup flowoperations
 * \brief Pick batch element.
 * \details Pick batch element from a batched expression.
 * \param x The input expression
 * \param v A pointer to the index of the correct element to be picked.
 *
 * \return The expression of picked batch element. The picked element is a tensor
 *         whose `bd` equals to one.
 */
    inline Expression pick_batch_elem(const Expression &x, const unsigned *v) { return dynet::pick_batch_elem(x, v); }

/**
 * \ingroup flowoperations
 * \brief Pick batch elements.
 * \details Pick several batch elements from a batched expression.
 * \param x The input expression
 * \param v A pointer to the indexes
 *
 * \return The expression of picked batch elements. The batch elements is a tensor
 *         whose `bd` equals to the size of vector `v`.
 */
    inline Expression
    pick_batch_elems(const Expression &x, const std::vector<unsigned> *pv) { return dynet::pick_batch_elems(x, pv); }

/**
 * \ingroup flowoperations
 * \brief Concatenate list of expressions to a single batched expression
 * \details Perform a concatenation of several expressions along the batch dimension.
 *          All expressions must have the same shape except for the batch dimension.
 *
 * \param xs The input expressions
 *
 * \return The expression with the batch dimensions concatenated
 */
    inline Expression concatenate_to_batch(const std::vector<Expression> &xs) {
      return dynet::concatenate_to_batch(Expression::vector_cast_to_base(xs));
    }

/**
 * \ingroup flowoperations
 * \brief Strided select in multiple dimensions
 * \details Select a range and/or stride of elements from an expression.
 *
 * \param x The input expression
 * \param strides List of strides for each dimension, must be >= 1. Dimensions not included default to 1. Batch dimension can be included as very last dimension.
 * \param from    List of 0-based offsets (inclusive) for each dimension, must be >= 0. Dimensions not included default to 0. Batch dimension can be included as very last dimension.
 * \param to      List of highest 0-based index to select (exclusive) for each dimension, must be >= 0. Dimensions not included default to the corresponding dim size. Batch dimension can be included as very last dimension.
 *
 * \return The value of x[from[0]:to[0]:strides[0],..] (as it would be in numpy syntax)
 */
    inline Expression
    strided_select(const Expression &x, const std::vector<int> &strides, const std::vector<int> &from = {},
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
    inline Expression concatenate_cols(const std::vector<Expression> &xs) { return dynet::concatenate_cols(Expression::vector_cast_to_base(xs)); }

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
    inline Expression
    concatenate(const std::initializer_list<Expression> &xs, unsigned d = 0) { return dynet::concatenate(Expression::vector_cast_to_base(xs), d); }

    inline Expression concatenate(const std::vector<Expression> &xs, unsigned d = 0) {
      return dynet::concatenate(Expression::vector_cast_to_base(xs), d);
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
    inline Expression max_dim(const Expression &x, unsigned d = 0) { return dynet::max_dim(x, d); }

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
    inline Expression min_dim(const Expression &x, unsigned d = 0) { return dynet::min_dim(x, d); }


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
    inline Expression noise(const Expression &x, float stddev) { return dynet::noise(x, stddev); }

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
    inline Expression dropout(const Expression &x, float p) { return dynet::dropout(x, p); }

/**
 * \ingroup noiseoperations
 * \brief Dropout along a specific dimension
 * \details Identical to the dropout operation except the dropout mask is the same across one dimension. Use this if you want to drop columns or lines in a matrix for example
 *
 * For now this only supports tensors of order <= 3 (with or without batch dimension)
 *
 * \param x The input expression
 * \param d The dimension along which to drop
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
    inline Expression dropout_dim(const Expression &x, unsigned d, float p) { return dynet::dropout_dim(x, d, p); }

/**
 * \ingroup noiseoperations
 * \brief Dropout entire elements of a minibatch
 * \details Identical to the dropout operation except entire batch elements are dropped
 *
 * \param x The input expression
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
    inline Expression dropout_batch(const Expression &x, float p) { return dynet::dropout_batch(x, p); }

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
    inline Expression block_dropout(const Expression &x, float p) { return dynet::block_dropout(x, p); }


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
    inline Expression contract3d_1d(const Expression &x, const Expression &y) { return dynet::contract3d_1d(x, y); }
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
    inline Expression contract3d_1d_1d(const Expression &x, const Expression &y, const Expression &z) {
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
    inline Expression contract3d_1d_1d(const Expression &x, const Expression &y, const Expression &z,
                                       const Expression &b) { return dynet::contract3d_1d_1d(x, y, z, b); }
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
    inline Expression contract3d_1d(const Expression &x, const Expression &y, const Expression &b) {
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
    inline Expression inverse(const Expression &x) { return dynet::inverse(x); }

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
    inline Expression logdet(const Expression &x) { return dynet::logdet(x); }

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
    inline Expression trace_of_product(const Expression &x, const Expression &y) {
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
 * \param x Input expression (possibly batched)
 * \param g Gain (same dimension as x, no batch dimension)
 * \param b Bias (same dimension as x, no batch dimension)
 * \return An expression of the same dimension as `x`
 */
    inline Expression
    layer_norm(const Expression &x, const Expression &g, const Expression &b) { return dynet::layer_norm(x, g, b); }

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
    inline Expression weight_norm(const Expression &w, const Expression &g) { return dynet::weight_norm(w, g); }
  }
}

#endif //DYNET_WRAPPER_DY_OPERATIONS_HPP
