#include "Settings.hpp"
#include <boost/lexical_cast.hpp>
#include <ftxui/component/captured_mouse.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/loop.hpp>

using namespace ftxui;

Component wrapFormElement(std::string label, Component& c){
    return Renderer(c, [&] {
        return hbox({
            text(label) | size(WIDTH, EQUAL, 30),
            separator(),
            c->Render() | xflex,
        }) | xflex;
    });
}

Settings::Settings(){
    auto screen = ScreenInteractive::FitComponent();

    bool submitted = false;
    std::string errorMessage = "";

    int usingLG = 0;
    std::vector<std::string> rateMatrixChoices = {
        "Use LG Rate Matrix",
        "Use GTR Rate Matrix"
    };
    auto rateMatrixBoxes = Radiobox(&rateMatrixChoices, &usingLG);
    auto rateMatrixInput =  wrapFormElement("Rate Matrices", rateMatrixBoxes);

    auto invarBox = Checkbox("", &invar);
    auto invarInput = wrapFormElement("Invariant Sites", invarBox);

    std::string numRatesString = std::to_string(numRates);
    auto ratesField = Input(&numRatesString, "4");
    auto ratesInput = wrapFormElement("Rate Number", ratesField);

    std::string numThreadString = std::to_string(numThreads);
    auto threadField = Input(&numThreadString, "4");
    auto threadInput = wrapFormElement("Thread Number", threadField);

    std::string rejuvenationString = std::to_string(rejuvenationIterations);
    auto rejuvenationField = Input(&rejuvenationString, "10");
    auto rejuvenationInput = wrapFormElement("MCMC Iterations", rejuvenationField);

    std::string numParticleString = std::to_string(numParticles);
    auto particleField = Input(&numParticleString, "500");
    auto particleInput = wrapFormElement("Particle Number", particleField);

    std::string seedString = std::to_string(seed);
    auto seedField = Input(&seedString, "1");
    auto seedInput = wrapFormElement("RNG Seed", seedField);

    auto inputField = Input(&fastaFile, "Your FASTA");
    auto fileInput = wrapFormElement("Input File", inputField);
    
    auto onSumbit = [&] {
        if(fastaFile.empty()){
            errorMessage = "No FASTA provided";
            return;
        }

        try {
            seed = boost::lexical_cast<unsigned int>(seedString);
        }
        catch(...) {
            errorMessage = "Seed is an unsigned int";
            return;
        }

        try {
            numParticles = boost::lexical_cast<int>(numParticleString);
        }
        catch(...) {
            errorMessage = "Particle number is an int";
            return;
        }

        try {
            numThreads = boost::lexical_cast<int>(numThreadString);
        }
        catch(...) {
            errorMessage = "Thread number is an int";
            return;
        }

        try {
            numRates = boost::lexical_cast<int>(numRatesString);
        }
        catch(...) {
            errorMessage = "Rate number is an int";
            return;
        }
          
        try {
            rejuvenationIterations = boost::lexical_cast<int>(rejuvenationString);
        }
        catch(...) {
            errorMessage = "Rejuvenation iterations is an int";
            return;
        }

        lg = (usingLG == 0);

        submitted = true;
        screen.Exit();
    };
    auto exitButton = Button("Save and Run", onSumbit);


    auto layout = Container::Vertical({
        rateMatrixInput,
        invarInput,
        ratesInput,
        threadInput,
        rejuvenationInput,
        particleInput,
        seedInput,
        fileInput,
        exitButton
    });

    // Wrap visual rendering
    auto renderer = Renderer(layout, [&] {
        Elements catElements = {
            text(""),
            text(""),
            text(""),
            text("     /\\_____/\\   "),
            text("    /  o   o  \\   "),
            text("   ( ==  ^  == )   "),
            text("    )         (    "),
            text("   (           )   "),
            text("  ( (  )   (  ) )  "),
            text(" (__(__)___(__)__) "),
            separator(),
        };

        if(!errorMessage.empty()){
            catElements.push_back(
                text(errorMessage) | color(Color::Red) | bold
            );
        }

        return hbox({
            vbox({
                catElements
            }) | border,
            vbox({
                text("FastCAT Interactive Menu") | bold,
                separator(),
                rateMatrixInput->Render(),
                invarInput->Render(),
                ratesInput->Render(),
                threadInput->Render(),
                rejuvenationInput->Render(),
                particleInput->Render(),
                seedInput->Render(),
                fileInput->Render(),
                separator(),
                exitButton->Render() | xflex_grow
            }) | xflex | size(WIDTH, GREATER_THAN, 50) | border
        });
    });

    screen.Loop(renderer);

    if(!submitted)
        std::exit(0);
}

Settings::Settings(int argc, char* argv[]){

    std::vector<std::string> arguments;
    for(int i = 1; i < argc; i++){
        std::string arg = argv[i];
        arguments.push_back(arg);
    }

    bool chosenMatrix = false; // We want the user to be forced to set this
    bool setFasta = false;

    std::string lastArgument;
    for(const auto& arg : arguments){
        
        if(arg == "-LG"){
            if(chosenMatrix){
                usage();
                std::cout << "Error: Multiple rate matrices have been selected! Are you sure you've set your arguments correctly?" << std::endl;
                std::exit(1);
            }

            lg = true;
            chosenMatrix = true;
        }
        else if(arg == "-GTR"){
            if(chosenMatrix){
                usage();
                std::cout << "Error: Multiple rate matrices have been selected! Are you sure you've set your arguments correctly?" << std::endl;
                std::exit(1);
            }
            
            lg = false;
            chosenMatrix = true;
        }
        else if(arg == "-I"){
            invar = true;
        }
        else if(arg == "-h" || arg == "-help"){
            usage();
            std::exit(0);
        }
        else if(lastArgument == "-G"){
            try {
                numRates = boost::lexical_cast<int>(arg);
            }
            catch(...) {
                usage();
                std::cout << "Error: -G is supposed to be an integer representing the number of discretized rate categories." << std::endl;
                std::exit(1);
            }
        }
        else if(lastArgument == "-t"){
            try {
                numThreads = boost::lexical_cast<int>(arg);
            }
            catch(...) {
                usage();
                std::cout << "Error: -t is supposed to be an integer representing the number of threads." << std::endl;
                std::exit(1);
            }
        }
        else if(lastArgument == "-p"){
            try {
                numParticles = boost::lexical_cast<int>(arg);
            }
            catch(...) {
                usage();
                std::cout << "Error: -p is supposed to be an integer representing the number of particles." << std::endl;
                std::exit(1);
            }
        }
        else if(lastArgument == "-r"){
            try {
                rejuvenationIterations = boost::lexical_cast<int>(arg);
            }
            catch(...) {
                usage();
                std::cout << "Error: -r is supposed to be an integer representing the number of rejuvenation iterations." << std::endl;
                std::exit(1);
            }
        }
        else if(lastArgument == "-s"){
            try {
                seed = boost::lexical_cast<unsigned int>(arg);
            }
            catch(...) {
                usage();
                std::cout << "Error: -s is supposed to be a numeric (unsigned int) seed." << std::endl;
                std::exit(1);
            }
        }
        else if(lastArgument == "-a"){
            fastaFile = arg;
            setFasta = true;
        }
        else{
            usage();
            std::cout << "Error: Unrecognized argument " << arg << std::endl;
            std::exit(1);
        }

        lastArgument = arg;
    }


    if(!chosenMatrix){
        usage();
        std::cout << "Error: No rate matrix has been chosen! Please set -GTR or -LG." << std::endl;
        std::exit(1);
    }

    if(!setFasta){
        usage();
        std::cout << "Error: No alignment has been provided! Please provide a FASTA file with -a" << std::endl;
        std::exit(1);
    }
}

void Settings::usage(){
    std::cout << "Minimum FastCAT Usage:\n";
    std::cout << "\tfastcat -a <fasta_file> -LG\n";
    std::cout << "\tfastcat -a <fasta_file> -GTR\n";
    std::cout << "FastCAT Command Line Arguments:\n";
    std::cout << "\t-a <fasta_file>         Specifies the alignment for the analysis.\n";
    std::cout << "\t-t <num_threads>        Specify the number of threads to dedicate to the analysis.\n";
    std::cout << "\t-p <num_particles>      Specify the number of particles to use during SMC.\n";
    std::cout << "\t-r <num_iter>           Specify the number of rejuvenation iterations to use during SMC.\n";
    std::cout << "\t-s <seed>               Specify the RNG seed to use for your analysis. This is important to set if you are doing multiple analyses!\n";
    std::cout << "\t-LG                     Use the LG rate matrix.\n";
    std::cout << "\t-GTR                    Infer an amino acid GTR matrix.\n";
    std::cout << "\t-I                      Use invariant sites mixture.\n";
    std::cout << "\t-G <num_rates>          Specify the number of discretized rate categories in the rate mixture.\n";

    std::cout << std::flush;
}
