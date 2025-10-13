#ifndef MATH_HPP
#define MATH_HPP

namespace Math {
    double alnorm ( double x, bool upper );
    void chiSquareCDF ( int *n_data, int *a, double *x, double *fx );
    double gammad ( double x, double p, int *ifault );
    double ppchi2 ( double p, double v, double g, int *ifault );
    double ppnd ( double p, int *ifault );
    double r8Min ( double x, double y );
    double gammaLogPDF(double x, double shape, double rate);
    std::vector<double> getGammaDiscretization(int numBins, double alpha);
    double sampleBeta(double alpha, double beta, std::mt19937& gen);
    double betaLogPDF(double x, double alpha, double beta);
    double stationaryDirichletLogPDF(const Eigen::Vector<double, 20>& x, const Eigen::Vector<double, 20>& alpha);
}

#endif