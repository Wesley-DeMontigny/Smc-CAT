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
            text(label) | size(WIDTH, EQUAL, 25),
            separator(),
            c->Render() | xflex,
        }) | xflex;
    });
}

Settings::Settings(){
    auto screen = ScreenInteractive::FitComponent();

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
    auto ratesInput = wrapFormElement("Discretized Rates", ratesField);

    std::string numThreadString = std::to_string(numThreads);
    auto threadField = Input(&numThreadString, "4");
    auto threadInput = wrapFormElement("Thread Number", threadField);

    std::string rejuvenationString = std::to_string(rejuvenationIterations);
    auto rejuvenationField = Input(&rejuvenationString, "10");
    auto rejuvenationInput = wrapFormElement("Rejuvenation Iterations", rejuvenationField);

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
            seed = boost::lexical_cast<int>(seedString);
        }
        catch(...) {
            errorMessage = "Seed is an int";
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
                paragraph(errorMessage) | color(Color::Red) | bold
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

    lg = (usingLG == 0);
}

Settings::Settings(int argc, char* argv[]){
    // Implement command line reading
}