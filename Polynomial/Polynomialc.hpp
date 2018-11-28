/*
 * Copyright (c) 2015, Jonathan Ventura
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/Polynomials>
#include <vector>

#include <Polynomial/PolynomialInternal.hpp>

namespace Polynomial
{
    /**
     * Polynomial
     *
     * Templated class for representing a polynomial expression.
     *
     * The template parameter indicates the degree of polynomial.  An n-th degree polynomial has n+1 coefficients.
     *
     * The coefficients are stored in this order:
     * c0 x^deg + c1 x^(deg-1) + ...
     *
     * A dynamically-sized version is available: Polynomial<Eigen::Dynamic>
     *
     * Note that the static and dynamic versions should not be mixed, i.e. do not add a dynamic polynomial to a static one.
     */
    template<int deg>
    class Polynomialc
    {
        Eigen::Matrix<double,deg+1,1> coef;
    public:
        Polynomialc()
        : coef( Eigen::Matrix<double,1,deg+1>::Zero() )
        {
            
        }
        
        Polynomialc( const Eigen::Matrix<double,deg+1,1> &coefin )
        : coef( coefin )
        {

        }
        
        Polynomialc( const Polynomialc<deg> &polyin )
        : coef( polyin.coef )
        {

        }
        
        Polynomialc( const double *coefin )
        : coef( Internal::vecmap<deg+1>( coefin ) )
        {
            
        }
        
        const Eigen::Matrix<double,deg+1,1> &coefficients() const
        {
            return coef;
        }

        Eigen::Matrix<double,deg+1,1> &coefficients()
        {
            return coef;
        }

        template<int degin>
        Polynomialc<Internal::max<degin,deg>::value> operator+(const Polynomialc<degin> &poly) const
        {
            Polynomialc<Internal::max<degin,deg>::value> p;
            p.coefficients().tail(degin+1) = poly.coefficients();
            p.coefficients().tail(deg+1) += coef;
            return p;
        }
        
        template<int degin>
        Polynomialc<Internal::max<degin,deg>::value> operator-(const Polynomialc<degin> &poly) const
        {
            Polynomialc<Internal::max<degin,deg>::value> p;
            p.coefficients().tail(deg+1) = coef;
            p.coefficients().tail(degin+1) -= poly.coefficients();
            return p;
        }
        
        template<int degin>
        Polynomialc<degin+deg> operator*(const Polynomialc<degin> &poly) const
        {
            Polynomialc<degin+deg> p;
            Internal::PolyConv<deg,degin>::compute(p.coefficients(),coef,poly.coefficients());
            return p;
        }
        
        Polynomialc<deg> operator*(const double c) const
        {
            return Polynomialc<deg>(coef*c);
        }
        
        double eval(double x) const
        {
            return Internal::PolyVal<deg>::compute(coef,x);
        }
        
        void realRoots(std::vector<double> &roots) const
        {
            if ( coef[0] == 0 )
            {
                Internal::RootFinder<deg-1>::compute(coef.tail(deg),roots);
            } else {
                Internal::RootFinder<deg>::compute(coef,roots);
            }
        }
        
        void realRootsSturm(const double lb, const double ub, std::vector<double> &roots) const
        {
            if ( coef[0] == 0 )
            {
                Internal::SturmRootFinder<deg-1> sturm( coef.tail(deg) );
                sturm.realRoots( lb, ub, roots );
            } else {
                Internal::SturmRootFinder<deg> sturm( coef );
                sturm.realRoots( lb, ub, roots );
            }
        }
        
        void rootBounds( double &lb, double &ub )
        {
            Eigen::Matrix<double,deg,1> mycoef = coef.tail(deg).array().abs();
            mycoef /= fabs(coef(0));
            mycoef(0) += 1.;
            ub = mycoef.maxCoeff();
            lb = -ub;
        }
    };
    
    template <>
    class Polynomialc<Eigen::Dynamic>
    {
        Eigen::VectorXd coef;
    public:
        Polynomialc(const int deg)
        : coef( Eigen::VectorXd::Zero(deg+1) )
        {
            
        }
        
        Polynomialc( const Eigen::VectorXd &coefin)
        : coef( coefin )
        {

        }
        
        Polynomialc(const Polynomialc<Eigen::Dynamic> &polyin)
        : coef( polyin.coef )
        {

        }
        
        const Eigen::VectorXd &coefficients() const
        {
            return coef;
        }
        
        Eigen::VectorXd &coefficients()
        {
            return coef;
        }
        
        Polynomialc<Eigen::Dynamic> operator+(const Polynomialc<Eigen::Dynamic> &poly) const
        {
            int deg = coef.rows()-1;
            int degin = poly.coef.rows()-1;
            Polynomialc<Eigen::Dynamic> p( std::max(deg,degin) );
            p.coef.tail(degin+1) = poly.coef;
            p.coef.tail(deg+1) += coef;
            return p;
        }

        Polynomialc<Eigen::Dynamic> operator-(const Polynomialc<Eigen::Dynamic> &poly) const
        {
            int deg = coef.rows()-1;
            int degin = poly.coef.rows()-1;
            Polynomialc<Eigen::Dynamic> p( std::max(deg,degin) );
            p.coef.tail(deg+1) = coef;
            p.coef.tail(degin+1) -= poly.coef;
            return p;
        }
        
        Polynomialc<Eigen::Dynamic> operator*(const Polynomialc<Eigen::Dynamic> &poly) const
        {
            int deg = coef.rows()-1;
            int degin = poly.coef.rows()-1;
            Polynomialc<Eigen::Dynamic> p( deg+degin );
            Internal::PolyConv<Eigen::Dynamic,Eigen::Dynamic>::compute(p.coef,coef,poly.coef);
            return p;
        }
        
        Polynomialc<Eigen::Dynamic> operator*(const double c) const
        {
            return Polynomialc<Eigen::Dynamic>(coef*c);
        }
        
        double eval(double x) const
        {
            return Internal::PolyVal<Eigen::Dynamic>::compute(coef,x);
        }
        
        void realRoots(std::vector<double> &roots) const
        {
            if ( coef[0] == 0 )
            {
                int deg = coef.rows()-1;
                Internal::RootFinder<Eigen::Dynamic>::compute(coef.tail(deg),roots);
            } else {
                Internal::RootFinder<Eigen::Dynamic>::compute(coef,roots);
            }
        }
        
        void realRootsSturm(const double lb, const double ub, std::vector<double> &roots) const
        {
            if ( coef[0] == 0 )
            {
                int deg = coef.rows()-1;
                Internal::SturmRootFinder<Eigen::Dynamic> sturm( coef.tail(deg) );
                sturm.realRoots( lb, ub, roots );
            } else {
                Internal::SturmRootFinder<Eigen::Dynamic> sturm( coef );
                sturm.realRoots( lb, ub, roots );
            }
        }
        
        void rootBounds( double &lb, double &ub )
        {
            int deg = coef.rows()-1;
            Eigen::VectorXd mycoef = coef.tail(deg).array().abs()/fabs(coef(0));
            mycoef(0) += 1.;
            ub = mycoef.maxCoeff();
            lb = -ub;
        }
    };
    
} // end namespace Polynomialc

#endif

