#include "analysis/Mcmc.hpp"
#include "analysis/Particle.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/SerializedParticle.hpp"
#include "core/Alignment.hpp"
#include "core/Miscellaneous.hpp"
#include "core/Settings.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <unordered_map>

int main(int argc, char* argv[]){
    #ifdef USE_UI
    Settings settings = (argc <= 1)
                        ? Settings()
                        : Settings(argc, argv);
    #else
    Settings settings = Settings(argc, argv);
    #endif

    if(settings.numThreads > omp_get_max_threads()){
        std::cout << "Error: The number of threads requested is larger than the number of maximum threads available (" << omp_get_max_threads() << ")!" << std::endl;
        std::exit(1);
    }

    omp_set_num_threads(settings.numThreads);

    boost::random::mt19937 rng{settings.seed};
    boost::random::uniform_01<double> unif{};
    
    std::vector<double> rawLogLikelihoods{};
    std::vector<double> rawWeights{};
    std::vector<double> normalizedWeights{};

    std::vector<std::set<boost::dynamic_bitset<>>> particleSplits;
    std::vector<SerializedParticle> oldSerializedParticles;
    std::vector<SerializedParticle> currentSerializedParticles;

    Alignment aln(settings.fastaFile);
    int numChar = aln.getNumChar();

    std::unordered_map<boost::dynamic_bitset<>, double> splitPosteriorProbabilities;

    std::cout << "Utilizing " << settings.numThreads << " threads for working with " << settings.numParticles << std::endl;

    Mcmc mcmc{};
    mcmc.emplaceMove(std::tuple{
        2.0,
        [&aln](Particle& m){
            return static_cast<int>(aln.getNumTaxa() / 2.0);
        },
        [](Particle& m){
            return m.branchMove();
        }
    });
    mcmc.emplaceMove(std::tuple{
        2.0,
        [&aln](Particle& m){
            return static_cast<int>(aln.getNumTaxa() / 2.0);
        },
        [&splitPosteriorProbabilities](Particle& m){
            return m.topologyMove(splitPosteriorProbabilities);
        }
    });
    mcmc.emplaceMove(std::tuple{
        1.0,
        [](Particle& m){
            return m.getNumCategories() * 5;
        },
        [](Particle& m){
            return m.stationaryMove();
        }
    });
    if(settings.numRates > 1){
        mcmc.emplaceMove(std::tuple{
            1.0,
            [](Particle& m){
                return 5;
            },
            [](Particle& m){
                return m.shapeMove();
            }
        });
    }
    if(settings.invar){
        mcmc.emplaceMove(std::tuple{
            1.0,
            [](Particle& m){
                return 5;
            },
            [](Particle& m){
                return m.invarMove();
            }
        });
    }
    if(!settings.lg){
        mcmc.emplaceMove(std::tuple{
            1.0,
            [](Particle& m){
                return 30;
            },
            [](Particle& m){
                return m.rateMatrixMove();
            }
        });
    }
    mcmc.initMoveProbs();

    auto rateMatrixCoords = RateMatrices::contructLowerTriangleCoordinates();

    std::vector<Particle> particles;

    std::chrono::steady_clock::time_point preAnalysis = std::chrono::steady_clock::now();

    particles.reserve(settings.numThreads);
    for (int t = 0; t < settings.numThreads; t++) {
        particles.emplace_back(settings.seed + t, aln, settings.numRates, settings.invar, settings.lg);
    }

    std::cout << "Initializing SMC..." << std::endl;

    rawLogLikelihoods.reserve(settings.numParticles);
    rawWeights.reserve(settings.numParticles);
    normalizedWeights.reserve(settings.numParticles);
    particleSplits.reserve(settings.numParticles);
    for(int i = 0; i < settings.numParticles; i++){
        particleSplits.push_back({});
        rawLogLikelihoods.push_back(0.0);
        rawWeights.push_back(-1.0 * std::log(settings.numParticles));
        normalizedWeights.push_back(0.0);
    }

    currentSerializedParticles.reserve(settings.numParticles);
    oldSerializedParticles.reserve(settings.numParticles);
    Particle& initP = particles[0];
    for(int n = 0; n < settings.numParticles; n++){  
        rawLogLikelihoods[n] = initP.lnLikelihood();
        SerializedParticle newParticle{};
        initP.writeToSerialized(newParticle);
        currentSerializedParticles.emplace_back(std::move(newParticle));
        particleSplits[n] = initP.getSplitSet();
        initP.initialize(settings.invar);
    }
    oldSerializedParticles = currentSerializedParticles;
    std::cout << "Initialized Particles..." << std::endl;

    std::array<double, 10> moveCount{};
    std::array<double, 10> acceptCount{};
    
    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    double rejThreshold = settings.rejuvenationThreshold * settings.numParticles;
    double lastTemp = 0.0;
    for(int iteration = 1; lastTemp < 1.0; iteration++){
        double currentTemp = lastTemp;
        double ESS = 0.0;
        computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, rejThreshold, currentTemp, lastTemp);

        std::cout << iteration << "\tTemp: " << currentTemp << "\tESS: " << ESS << std::endl;

        if(std::isnan(ESS)){
            std::cout << "Error: We have run into a NaN ESS! This usually happens when something has failed when computing the likelihoods." << std::endl;
            for(int l = 0; l < rawLogLikelihoods.size(); l++){
                if(std::isnan(rawLogLikelihoods[l])){
                    auto& p = currentSerializedParticles[l];
                    std::cout << "Found a NaN:          " << l << std::endl;
                    std::cout << "  Shape:              " << p.shape << std::endl;
                    std::cout << "  Tree:               " << p.newick << std::endl;
                    std::cout << "  pInvar:             " << p.pInvar << std::endl;
                    std::cout << "  Rate Matrix:        " << p.baseMatrix << std::endl;
                    std::cout << "  Stationary Dists    : " << std::endl;
                    for(auto& s : p.stationaries)
                        std::cout << s << std::endl;
                }
            }
            std::exit(1);
        }

        if(ESS <= rejThreshold && currentTemp != 1.0){
            std::cout << "Resampling Particles..." << std::endl;

            // Systematic resampling
            std::vector<int> assignments;
            assignments.reserve(settings.numParticles);
            double cumulative = normalizedWeights[0];
            int k = 0;
            double ruler = unif(rng) / static_cast<double>(settings.numParticles);
            double step = 1.0 / static_cast<double>(settings.numParticles);
            while(static_cast<int>(assignments.size()) < settings.numParticles){
                if(ruler <= cumulative){
                    assignments.push_back(k);
                    ruler += step;
                } 
                else{
                    if (++k >= settings.numParticles) { k = settings.numParticles - 1; cumulative = 1.0; }
                    else cumulative += normalizedWeights[k];
                }
            }

            std::cout << "Tuning MCMC..." << std::endl;
            using VarAcc = boost::accumulators::accumulator_set<
                double,
                boost::accumulators::stats<boost::accumulators::tag::variance>
            >;

            computeSplitPosteriors(
                splitPosteriorProbabilities,
                assignments,
                particleSplits
            );

            VarAcc rateVar{};
            double shapeDelta = 0.5;

            VarAcc branchVar{};
            double scaleDelta = 1.0;

            VarAcc stationaryVar{};
            double stationaryDelta = 1.0;

            double pInvarMean = 0.0;
            double invarAlpha = 0.5;

            Eigen::MatrixXd rateSampleMatrix(settings.numParticles, 190);
            Eigen::MatrixXd rateMatrixCovariance(190, 190);
            Eigen::MatrixXd rateMatrixCholesky{};
            double shrinkageDiagonal = 0.0;

            int index = 0;
            for(const auto& p : assignments){
                auto& particle = currentSerializedParticles[p];

                rateVar(std::log(particle.shape));
                pInvarMean += particle.pInvar;

                for(const auto& l : particle.branchLengths){
                    branchVar(std::log(l));
                }

                for(const auto& s : particle.stationaries){
                    for(int d = 0; d < 20; d++){
                        stationaryVar(s[d]);
                    }
                }

                if(!settings.lg){
                    for(int d = 0; d < 190; d++){
                        rateSampleMatrix(index, d) = particle.baseMatrix(d);
                    }
                }

                index++;
            }
            if(!settings.lg){
                rateMatrixCovariance = rateSampleMatrix.transpose() * rateSampleMatrix;
                rateMatrixCovariance *= (2.38 * 2.38) / 190.0; // Roberts, Gelman & Gilks
                rateMatrixCovariance *= 0.21; // Make the proposal just a little bit more conservative
                double shrinkageFactor = 0.2;
                for(int d = 0; d < 190; d++){
                    shrinkageDiagonal += rateMatrixCovariance(d,d);
                }
                shrinkageDiagonal /= 190.0;
                shrinkageDiagonal *= shrinkageFactor;
                rateMatrixCovariance *= 1.0 - shrinkageFactor;

                for(int d = 0; d < 190; d++){
                    rateMatrixCovariance(d,d) += shrinkageDiagonal; // Apply shrinkage to \lambda I v
                    rateMatrixCovariance(d,d) += 1e-6; // For stablibility
                }

                Eigen::LLT<Eigen::MatrixXd> chol(rateMatrixCovariance);
                if(chol.info() != Eigen::Success) {
                    std::cout << rateMatrixCovariance << std::endl;
                    std::cout << "Failed to Form Cholesky Factors for Rate Matrix Covariance!" << std::endl;
                    std::exit(1);
                }
                rateMatrixCholesky = chol.matrixL();
            }

            pInvarMean /= static_cast<double>(assignments.size());
            shapeDelta *= boost::accumulators::variance(rateVar);
            scaleDelta *= boost::accumulators::variance(branchVar);
            stationaryDelta *= boost::accumulators::variance(stationaryVar);

            for(auto& p : particles){
                p.shapeDelta = std::max(shapeDelta, 0.01);
                p.scaleDelta = std::max(scaleDelta, 0.01);
                p.subtreeScaleDelta = std::max(scaleDelta, 0.01);
                p.stationaryDelta = std::max(stationaryDelta, 0.01);
                p.invarAlpha = invarAlpha;
                if(!settings.lg){
                    p.rateMatrixDelta = std::max(shrinkageDiagonal, 0.01);
                    p.rateMatrixCholesky = rateMatrixCholesky;
                }
            }

            // Mutate particles after resampling
            std::cout << "Rejuvenating Particles..." << std::endl;
            #pragma omp parallel for schedule(dynamic)
            for(int n = 0; n < settings.numParticles; n++){
                int threadID = omp_get_thread_num();
                Particle& p = particles[threadID];
                int particleID = assignments[n];
                p.copyFromSerialized(currentSerializedParticles[particleID]);

                mcmc(p, settings.rejuvenationIterations, currentTemp);
                if(unif(rng) < settings.alg8Probability){ // Update the CRP only for a fraction of particles
                    p.gibbsPartitionMove(currentTemp);
                    p.refreshLikelihood(true);
                }
                rawLogLikelihoods[n] = p.lnLikelihood();
                rawWeights[n] = -1.0 * std::log(settings.numParticles); // Technically this happens during resampling, but our MCMC kernel is invariant, so theres no difference
                p.writeToSerialized(oldSerializedParticles[n]);

                particleSplits[n] = p.getSplitSet();
            }
            std::swap(currentSerializedParticles, oldSerializedParticles);
           
            std::fill(moveCount.begin(), moveCount.end(), 0);
            std::fill(acceptCount.begin(), acceptCount.end(), 0);
            for(auto& p : particles){
                for(int m = 0; m < moveCount.size(); m++){
                    moveCount[m] += p.moveCount[m];
                    acceptCount[m] += p.acceptCount[m];
                }
            }

            for(int m = 0; m < moveCount.size(); m++){
                std::cout << MoveNames[m] << " Acceptance Rate: " << (double)acceptCount[m]/(double)moveCount[m];
                if(m != moveCount.size() - 1){
                    std::cout << "\t";
                }
                else{
                    std::cout << std::endl;
                }
            }
        }

        lastTemp = currentTemp;
    }

    computeSplitPosteriors(
        splitPosteriorProbabilities,
        normalizedWeights,
        particleSplits
    );

    std::cout << "Computing the Greedy Consensus Tree..." << std::endl;
    std::vector<std::pair<boost::dynamic_bitset<>, double>> sortedSplits(
        splitPosteriorProbabilities.begin(),
        splitPosteriorProbabilities.end()
    );
    std::sort(sortedSplits.begin(), sortedSplits.end(), [](auto& a, auto& b){
        return a.second > b.second;
    });

    // Greedily select the best splits until we have a resolved tree.
    std::vector<boost::dynamic_bitset<>> selectedSplits;
    for(auto& split : sortedSplits){
        bool compatible = true;
        for(auto& bitset : selectedSplits){
            if(!((split.first & bitset).none() ||
                (split.first & ~bitset).none() ||
                (~split.first & bitset).none() ||
                (~split.first & ~bitset).none())){ // Four point condition for split compatibility
                compatible = false;
                break;
            }
        }
        if(compatible){
            selectedSplits.push_back(split.first);
        }
    }
    // Sort these in terms of size of the split so that it can be in the correct order for the build algorithm
    std::sort(selectedSplits.begin(), selectedSplits.end(), [](auto& a, auto& b){
        return a.count() < b.count();
    });

    // Accumulate weighted conditional mean for branch length splits
    using Acc = boost::accumulators::accumulator_set<
        double,
        boost::accumulators::stats<boost::accumulators::tag::weighted_mean>,
        double
    >;
    std::unordered_map<boost::dynamic_bitset<>, Acc> branchWeightedMeans{};
    for(auto& s : selectedSplits){
        branchWeightedMeans.emplace(s, Acc{});
    }
    for (int n = 0; n < settings.numParticles; n++) {
        Particle& p = particles[0];
        p.copyFromSerialized(currentSerializedParticles[n]);
        std::unordered_map<boost::dynamic_bitset<>, double> sbMap = p.getSplitBranchMap();
        for(const auto& [split, bL] : sbMap){
            if(branchWeightedMeans.contains(split)){
                branchWeightedMeans.at(split)(bL, boost::accumulators::weight = normalizedWeights[n]);
            }
        }
    }
    // Construct inputs
    std::vector<std::pair<boost::dynamic_bitset<>, double>> buildInputs;
    buildInputs.reserve(selectedSplits.size());
    for(auto& s : selectedSplits){
        double weightedMean = boost::accumulators::weighted_mean(branchWeightedMeans.at(s));
        buildInputs.emplace_back(std::move(s), weightedMean); // Consume selected splits in the process
    }

    Tree consensusTree(buildInputs, aln.getTaxaNames());
    std::cout << consensusTree.generateNewick(splitPosteriorProbabilities) << std::endl;

    std::chrono::steady_clock::time_point postAnalysis = std::chrono::steady_clock::now();
    std::cout << "The analysis completed in " << std::chrono::duration_cast<std::chrono::minutes>(postAnalysis - preAnalysis).count() << "[minutes]" << std::endl;
    
    return 0;

}