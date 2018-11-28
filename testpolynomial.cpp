#include <iostream>
#include <vector>

#include "Polynomial/Polynomialc.hpp"

using Polynomial::Polynomialc;

int main( int argc, char **argv )
{
    Eigen::Matrix<double,6,1> acoeffs;
    acoeffs << 1,2,3,4,5,6;
    Polynomialc<5> a( acoeffs );
    Eigen::Matrix<double,5,1> bcoeffs;
    bcoeffs << 1,2,3,4,5;
    Polynomialc<4> b( bcoeffs );

    std::cout << "*** Polynomialc algebra test ***\n";
    
    std::cout << "a: [" << a.coefficients().transpose() << "]\n";
    std::cout << "b: [" << b.coefficients().transpose() << "]\n";
    
    std::cout << "a+b: [" << (a+b).coefficients().transpose() << "]\n";
    std::cout << "a-b: [" << (a-b).coefficients().transpose() << "]\n";
    
    std::cout << "a*b: [" << (a*b).coefficients().transpose() << "]\n";

    std::cout << "a(5): " << a.eval(5.f) << "\n";
    
    std::cout << "\n";

    std::cout << "*** Dynamically-sized polynomial algebra test ***\n";

    Eigen::VectorXd adcoeffs(6);
    adcoeffs << 1,2,3,4,5,6;
    Polynomialc<Eigen::Dynamic> ad( adcoeffs );
    Eigen::VectorXd bdcoeffs(5);
    bdcoeffs << 1,2,3,4,5;
    Polynomialc<Eigen::Dynamic> bd( bdcoeffs );
    
    std::cout << "a: [" << ad.coefficients().transpose() << "]\n";
    std::cout << "b: [" << bd.coefficients().transpose() << "]\n";

    std::cout << "a+b: [" << (ad+bd).coefficients().transpose() << "]\n";
    std::cout << "a-b: [" << (ad-bd).coefficients().transpose() << "]\n";
    
    std::cout << "a*b: [" << (ad*bd).coefficients().transpose() << "]\n";
    
    std::cout << "a(5): " << ad.eval(5.f) << "\n";
    
    std::cout << "\n";
    
    std::cout << "*** Polynomialc roots test ***\n";
    
    Eigen::Matrix<double,5,1> ccoeffs;
    ccoeffs <<    -0.8049,    -0.4430,    0.0938,    0.9150,    0.9298;
    Polynomialc<4> c(ccoeffs);
    
    std::cout << "c: [" << c.coefficients().transpose() << "]\n";
    
    double lb,ub;
    c.rootBounds(lb,ub);
    std::cout << "bounds on roots: " << lb << "  " << ub << "\n";
    
    std::vector<double> roots;
    c.realRoots(roots);
    std::cout << "roots using companion matrix:\n";
    for ( int i = 0; i < roots.size(); i++ ) std::cout << "c(" << roots[i] << "): " << c.eval(roots[i]) << "\n";

    roots.clear();
    c.realRootsSturm(lb,ub,roots);
    std::cout << "roots using sturm sequences:\n";
    for ( int i = 0; i < roots.size(); i++ ) std::cout << "c(" << roots[i] << "): " << c.eval(roots[i]) << "\n";
    
    std::cout << "\n";

    std::cout << "*** Dynamically-sized polynomial roots test ***\n";
    
    Eigen::VectorXd cdcoeffs(5);
    cdcoeffs <<    -0.8049,    -0.4430,    0.0938,    0.9150,    0.9298;
    Polynomialc<Eigen::Dynamic> cd(ccoeffs);
    
    std::cout << "c: [" << cd.coefficients().transpose() << "]\n";
    
    cd.rootBounds(lb,ub);
    std::cout << "bounds on roots: " << lb << "  " << ub << "\n";
    
    roots.clear();
    cd.realRoots(roots);
    std::cout << "roots using companion matrix:\n";
    for ( int i = 0; i < roots.size(); i++ ) std::cout << "c(" << roots[i] << "): " << c.eval(roots[i]) << "\n";
    
    roots.clear();
    cd.realRootsSturm(lb,ub,roots);
    std::cout << "roots using sturm sequences:\n";
    for ( int i = 0; i < roots.size(); i++ ) std::cout << "c(" << roots[i] << "): " << c.eval(roots[i]) << "\n";

    return 0;
}
