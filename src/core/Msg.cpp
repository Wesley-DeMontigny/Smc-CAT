#include <cstdlib>
#include <iostream>
#include "Msg.hpp"


/**
 * @brief Print an error to the screen and exit the program
 * @param s The error to be printed
 */
void Msg::error(std::string s) {

	std::cout << "Error: " << s << std::endl;
	std::cout << "Exiting program" << std::endl;
	std::exit(1);
}

/**
 * @brief Print a warning to the screen
 * @param s The warning to be printed
 */
void Msg::warning(std::string s) {

	std::cout << "Warning: " << s << std::endl;
}
