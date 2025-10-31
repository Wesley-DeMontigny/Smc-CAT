#include "analysis/Mcmc.hpp"
#include "analysis/Particle.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/SerializedParticle.hpp"
#include "core/Alignment.hpp"
#include "core/Settings.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <unordered_map>

void computeNextStep(std::vector<double>& rawWeights, std::vector<double>& rawLogLikelihoods, 
                     std::vector<double>& normalizedWeights, double& ESS,
                     double& currentTemp, const double& lastTemp, const int& numParticles){
    double targetESS = 0.6 * numParticles;
    double low = lastTemp;
    double high = 1.0;

    auto computeESS = [&](double temp){
        std::vector<double> tempWeights(numParticles);
        double total = 0.0;

        for(int n = 0; n < numParticles; n++){
            tempWeights[n] = rawWeights[n] + (temp - lastTemp) * rawLogLikelihoods[n];
        }

        double maxW = *std::max_element(tempWeights.begin(), tempWeights.end());

        std::vector<double> expWeights(numParticles);
        for(int n = 0; n < numParticles; n++){
            expWeights[n] = std::exp(tempWeights[n] - maxW);
            total += expWeights[n];
        }

        double sumSq = 0.0;
        for(int n = 0; n < numParticles; n++){
            expWeights[n] /= total;
            sumSq += expWeights[n] * expWeights[n];
        }

        return 1.0 / sumSq;
    };

    // Binary search for temperature
    while(high - low > 1e-5){
        double mid = 0.5 * (low + high);
        double midESS = computeESS(mid);

        if(midESS > targetESS){
            low = mid;   // We can still increase temperature
        } 
        else{
            high = mid;  // Too much degeneracy
        }
    }

    currentTemp = high;

    for (int n = 0; n < numParticles; n++){
        rawWeights[n] = rawWeights[n] + (currentTemp - lastTemp) * rawLogLikelihoods[n];
    }

    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());
    double total = 0.0;
    for (int n = 0; n < numParticles; n++){
        normalizedWeights[n] = std::exp(rawWeights[n] - maxW);
        total += normalizedWeights[n];
    }

    double sumSq = 0.0;
    for (int n = 0; n < numParticles; n++){
        normalizedWeights[n] /= total;
        sumSq += normalizedWeights[n] * normalizedWeights[n];
    }

    ESS = 1.0 / sumSq;
}

void computeSplitPosteriors(std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities, 
                            const std::vector<double>& normalizedWeights,
                            const std::vector<std::set<boost::dynamic_bitset<>>>& particleSplits,
                            const int& numParticles){
    splitPosteriorProbabilities.clear();
    for(int n = 0; n < numParticles; n++){
        for(boost::dynamic_bitset<> split : particleSplits[n]){
            if (splitPosteriorProbabilities.count(split)) {
                splitPosteriorProbabilities[split] += normalizedWeights[n];
            }
            else {
                splitPosteriorProbabilities[split] = normalizedWeights[n];
            }
        }
    }
}

int main(int argc, char** argv){
    Settings settings = (argc <= 1)
                        ? Settings()
                        : Settings(argc, argv);


    omp_set_num_threads(settings.numThreads);

    boost::random::mt19937 rng{settings.seed};
    boost::random::uniform_01<double> unif{};
    
    std::vector<double> rawLogLikelihoods(settings.numParticles, 0);
    std::vector<double> rawWeights(settings.numParticles, -1.0 * std::log(settings.numParticles));
    std::vector<double> normalizedWeights(settings.numParticles, 0.0);

    std::vector<std::set<boost::dynamic_bitset<>>> particleSplits;

    particleSplits.reserve(settings.numParticles);
    for(int i = 0; i < settings.numParticles; i++){
        particleSplits.push_back({});
    }

    std::vector<SerializedParticle> currentSerializedParticles;
    std::vector<SerializedParticle> oldSerializedParticles;
    currentSerializedParticles.reserve(settings.numParticles);
    oldSerializedParticles.reserve(settings.numParticles);

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
            return m.getNumCategories() * 2;
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
                return 10;
            },
            [](Particle& m){
                return m.rateMatrixMove();
            }
        });
    }
    mcmc.initMoveProbs();


    std::vector<Particle> particles;

    std::chrono::steady_clock::time_point preAnalysis = std::chrono::steady_clock::now();

    particles.reserve(settings.numThreads);
    for (int t = 0; t < settings.numThreads; t++) {
        particles.emplace_back(settings.seed + t, aln, settings.numRates, settings.invar, settings.lg);
    }

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    Particle& initP = particles[0];
    for(int n = 0; n < settings.numParticles; n++){  
        rawLogLikelihoods[n] = initP.lnLikelihood();
        currentSerializedParticles.emplace_back(
            initP.getNewick(), initP.getAssignments(), initP.getCategories(),
            initP.getBaseMatrix(), initP.getPInvar(), initP.getShape()
        );
        particleSplits[n] = initP.getSplitSet();
        initP.initialize(settings.invar);
    }
    oldSerializedParticles = currentSerializedParticles;
    std::cout << "Initialized Particles..." << std::endl;

    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    double lastTemp = 0.0;
    for(int iteration = 1; lastTemp < 1.0; iteration++){
        double currentTemp = lastTemp;
        double ESS = 0.0;
        computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, currentTemp, lastTemp, settings.numParticles);

        std::cout << iteration << "\tTemp: " << currentTemp << "\tESS: " << ESS << std::endl;

        computeSplitPosteriors(
            splitPosteriorProbabilities,
            normalizedWeights,
            particleSplits,
            settings.numParticles
        );

        if(ESS <= 0.6 * settings.numParticles && currentTemp != 1.0){
            std::cout << "Resampling Particles..." << std::endl;

            // Systematic resampling
            double u = unif(rng) / static_cast<double>(settings.numParticles);
            std::vector<int> assignments;
            assignments.reserve(settings.numParticles);
            double cumulative = normalizedWeights[0];
            int currentWeight = 0;
            int k = 0;
            double ruler = u;
            double step = 1.0 / settings.numParticles;
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

            // Mutate particles after resampling
            std::cout << "Rejuvenating Particles..." << std::endl;
            #pragma omp parallel for schedule(dynamic)
            for (int n = 0; n < settings.numParticles; n++) {
                int threadID = omp_get_thread_num();
                Particle& p = particles[threadID];
                int particleID = assignments[n];
                p.copyFromSerialized(currentSerializedParticles[particleID]);

                mcmc.run(p, settings.rejuvenationIterations, currentTemp);
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
        }

        lastTemp = currentTemp;
    }

    std::cout << "Computing the Majority Consensus Tree..." << std::endl;
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

    // Get mean branch lengths
    std::set<std::set<boost::dynamic_bitset<>>> deduped(
        particleSplits.begin(),
        particleSplits.end()
    );
    std::cout << "There were " << deduped.size() << " unique topologies in the final approximation" << std::endl;

    std::chrono::steady_clock::time_point postAnalysis = std::chrono::steady_clock::now();
    std::cout << "The analysis completed in " << std::chrono::duration_cast<std::chrono::minutes>(postAnalysis - preAnalysis).count() << "[minutes]" << std::endl;
    
    return 0;

}