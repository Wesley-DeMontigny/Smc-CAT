#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include "core/Settings.hpp"
#include "analysis/SerializedParticle.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Particle.hpp"
#include "core/Alignment.hpp"
#include "analysis/Mcmc.hpp"

void computeNextStep(std::vector<double>& rawWeights, std::vector<double>& rawLogLikelihoods, 
                     std::vector<double>& normalizedWeights, double& ESS,
                     double& currentTemp, const double& lastTemp, const int& numParticles) {
    double targetESS = 0.6 * numParticles;
    double low = lastTemp;
    double high = 1.0;

    auto computeESS = [&](double temp) {
        std::vector<double> tempWeights(numParticles);
        double total = 0.0;

        for (int n = 0; n < numParticles; n++) {
            tempWeights[n] = rawWeights[n] + (temp - lastTemp) * rawLogLikelihoods[n];
        }

        double maxW = *std::max_element(tempWeights.begin(), tempWeights.end());

        std::vector<double> expWeights(numParticles);
        for (int n = 0; n < numParticles; n++) {
            expWeights[n] = std::exp(tempWeights[n] - maxW);
            total += expWeights[n];
        }

        double sumSq = 0.0;
        for (int n = 0; n < numParticles; n++) {
            expWeights[n] /= total;
            sumSq += expWeights[n] * expWeights[n];
        }

        return 1.0 / sumSq;
    };

    // Binary search for temperature
    while (high - low > 1e-5) {
        double mid = 0.5 * (low + high);
        double midESS = computeESS(mid);

        if (midESS > targetESS) {
            low = mid;   // We can still increase temperature
        } else {
            high = mid;  // Too much degeneracy
        }
    }

    currentTemp = high;

    for (int n = 0; n < numParticles; n++) {
        rawWeights[n] = rawWeights[n] + (currentTemp - lastTemp) * rawLogLikelihoods[n];
    }

    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());
    double total = 0.0;
    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] = std::exp(rawWeights[n] - maxW);
        total += normalizedWeights[n];
    }

    double sumSq = 0.0;
    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] /= total;
        sumSq += normalizedWeights[n] * normalizedWeights[n];
    }

    ESS = 1.0 / sumSq;
}

int main(int argc, char** argv) {
    int numParticles = 500;
    int rejuvinationIterations = 10;
    int numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);
    bool invar = false;
    int numRates = 1;

    boost::random::mt19937 rng{1};
    boost::random::uniform_01<double> unif{};
    
    std::vector<double> rawLogLikelihoods(numParticles, 0);
    std::vector<double> rawWeights(numParticles, -1.0 * std::log(numParticles));
    std::vector<double> normalizedWeights(numParticles, 0.0);

    std::unordered_map<boost::dynamic_bitset<>, double> splitPosteriorProbabilities;
    std::vector<std::set<boost::dynamic_bitset<>>> particleSplits;
    particleSplits.reserve(numParticles);
    for(int i = 0; i < numParticles; i++){
        particleSplits.push_back({});
    }

    std::vector<SerializedParticle> currentSerializedParticles;
    std::vector<SerializedParticle> oldSerializedParticles;
    currentSerializedParticles.reserve(numParticles);
    oldSerializedParticles.reserve(numParticles);

    Alignment aln("/workspaces/FastCAT/local_testing/globin_aa_aligned.fasta");
    int numChar = aln.getNumChar();

    std::cout << "Utilizing " << numThreads << " threads for working with " << numParticles << std::endl;

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
    if(numRates > 1){
        mcmc.emplaceMove(std::tuple{
            1.0,
            [](Particle& m){
                return 3;
            },
            [](Particle& m){
                return m.shapeMove();
            }
        });
    }
    if(invar){
        mcmc.emplaceMove(std::tuple{
            1.0,
            [](Particle& m){
                return 3;
            },
            [](Particle& m){
                return m.invarMove();
            }
        });
    }
    mcmc.initMoveProbs();


    std::vector<Particle> particles;

    std::chrono::steady_clock::time_point preAnalysis = std::chrono::steady_clock::now();

    particles.reserve(numThreads);
    for (int t = 0; t < numThreads; t++) {
        particles.emplace_back(t, aln, numRates, invar);
    }

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    Particle& initP = particles[0];
    for(int n = 0; n < numParticles; n++){  
        rawLogLikelihoods[n] = initP.lnLikelihood();
        currentSerializedParticles.emplace_back(
            initP.getNewick(), initP.getAssignments(), initP.getCategories(),
            initP.getBaseMatrix(), initP.getPInvar(), initP.getShape()
        );
        particleSplits[n] = initP.getSplits();
        initP.initialize(invar);
    }
    oldSerializedParticles = currentSerializedParticles;
    std::cout << "Initialized Particles..." << std::endl;

    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    double lastTemp = 0.0;
    for(int i = 1; lastTemp < 1.0; i++){
        double currentTemp = lastTemp;
        double ESS = 0.0;
        computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, currentTemp, lastTemp, numParticles);

        // Generate split posterior probabilities
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

        std::cout << i << "\tTemp: " << currentTemp << "\tESS: " << ESS << std::endl;

        if(ESS <= 0.6 * numParticles && currentTemp != 1.0){
            std::cout << "Resampling Particles..." << std::endl;

            // Systematic resampling
            double u = unif(rng) / static_cast<double>(numParticles);
            std::vector<int> assignments;
            assignments.reserve(numParticles);
            double cumulative = normalizedWeights[0];
            int currentWeight = 0;
            int k = 0;
            double ruler = u;
            double step = 1.0 / numParticles;
            while (static_cast<int>(assignments.size()) < numParticles) {
                if (ruler <= cumulative) {
                    assignments.push_back(k);
                    ruler += step;
                } 
                else {
                    if (++k >= numParticles) { k = numParticles - 1; cumulative = 1.0; }
                    else cumulative += normalizedWeights[k];
                }
            }

            std::cout << "Rejuvenating Particles..." << std::endl;
            #pragma omp parallel for schedule(dynamic)
            for (int n = 0; n < numParticles; n++) {
                int threadID = omp_get_thread_num();
                Particle& p = particles[threadID];
                int particleID = assignments[n];
                p.copyFromSerialized(currentSerializedParticles[particleID]);

                mcmc.run(p, rejuvinationIterations, currentTemp);

                if(unif(rng) < 0.1){
                    p.gibbsPartitionMove(currentTemp);
                    p.refreshLikelihood(true);
                }
                rawLogLikelihoods[n] = p.lnLikelihood();
                rawWeights[n] = -1.0 * std::log(numParticles);
                p.writeToSerialized(oldSerializedParticles[n]);

                particleSplits[n] = p.getSplits();
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
        if(compatible){ // Anchor splits in orientation before pushing
            if (split.first.test(0))
                selectedSplits.push_back(split.first);
            else
                selectedSplits.push_back(~split.first);
        }
    }
    
    std::sort(selectedSplits.begin(), selectedSplits.end(), [](auto& a, auto& b){
        return a.count() < b.count();
    });

    auto consensusTree = Tree(selectedSplits, aln.getTaxaNames());
    std::vector<std::string> newickStrings;
    for(auto& particle : currentSerializedParticles){
        newickStrings.push_back(particle.newick);
    }
    consensusTree.assignMeanBranchLengths(newickStrings, normalizedWeights, aln.getTaxaNames());

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